"""Chunk world model for fast action-sequence scoring.

This module is intentionally separate from the denoising ``Dynamics`` model.
It follows the SimDist-style contract:

    context latents/actions + future action sequence -> future latent sequence

There is no sigma, no noisy future latent input, and no denoising schedule.
The model also exposes lightweight reward/continuation/value heads and a
future-action policy head so it can be trained as a self-contained chunk world
model while still using this repo's tanh-Gaussian BC loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model import (
        ActionEncoder,
        ContinuationHeadMTP,
        PolicyHeadMTP,
        RewardHeadMTP,
        StateEncoder,
        StatePredictorHead,
        TaskEmbedder,
        ValueHeadMTP,
        VectorHeadMTP,
        normalize_state_vec,
        sinusoid_table,
    )
except ImportError:  # Direct execution through the training scripts.
    from model import (
        ActionEncoder,
        ContinuationHeadMTP,
        PolicyHeadMTP,
        RewardHeadMTP,
        StateEncoder,
        StatePredictorHead,
        TaskEmbedder,
        ValueHeadMTP,
        VectorHeadMTP,
        normalize_state_vec,
        sinusoid_table,
    )


MODEL_ACTION_EPS = 1e-3
SCORER_ARCH_LEGACY = "legacy_hidden"
SCORER_ARCH_SIMDIST = "simdist_latent"
DYNAMICS_ARCH_SIMDIST = "simdist_action_decoder"
LATENT_CODEC_MLP = "mlp"
OBSERVATION_REPRESENTATION_TOKENIZER = "tokenizer"
OBSERVATION_REPRESENTATION_ROACH_PPO = "roach_ppo"
# CAT's MetaDrive benchmark: the 101-dim lidar/state vector, encoded by an MLP initialised from the expert TD3
# actor's first two layers (the counterpart of the ROACH PPO feature extractor)
OBSERVATION_REPRESENTATION_MD_VECTOR = "md_vector"
# The F1TENTH twin (mixed_quality_f1tenth): the PPO expert's 144-dim LiDAR/reference vector, encoded by an MLP
# initialised from the expert PPO actor's first two layers (tanh); otherwise handled like md_vector
OBSERVATION_REPRESENTATION_F1_VECTOR = "f1_vector"
VECTOR_REPRESENTATIONS = (OBSERVATION_REPRESENTATION_MD_VECTOR, OBSERVATION_REPRESENTATION_F1_VECTOR)
# What the five-output reward-factor head predicts: ValeoAction factors (CARLA), CAT's reward components
# (md_reward.py) or the F1TENTH reward factors (f1_reward.py). Only the output activation differs inside the model.
REWARD_FACTOR_KIND_VALEO = "valeo"
REWARD_FACTOR_KIND_CAT = "cat"
REWARD_FACTOR_KIND_F1TENTH = "f1tenth"


def remap_legacy_value_head_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a single-head checkpoint (``value_head.*``) onto the ensemble layout (``value_heads.0.*``)."""
    keys = list(state_dict.keys())
    if any(k.startswith("value_heads.") for k in keys) or not any(k.startswith("value_head.") for k in keys):
        return dict(state_dict)
    return {
        ("value_heads.0." + k[len("value_head."):] if k.startswith("value_head.") else k): v
        for k, v in state_dict.items()
    }


def remap_legacy_value_target_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a single target head state dict (``projector.*``) onto a ModuleList of heads (``0.projector.*``)."""
    keys = list(state_dict.keys())
    if not keys or all(k.split(".", 1)[0].isdigit() for k in keys):
        return dict(state_dict)
    return {"0." + k: v for k, v in state_dict.items()}


def value_ensemble_logits(
    heads: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every value head on ``features`` (B,T,D).

    Returns logits (B,T,K,bins) and the shared symlog bin centres (bins,). A single head (an
    ``nn.Module`` that is not a container) is treated as an ensemble of one.
    """
    if isinstance(heads, nn.Module) and not isinstance(heads, (nn.ModuleList, nn.Sequential)):
        heads = [heads]
    logits, centers = [], None
    for head in heads:
        head_logits, centers = head(features)
        logits.append(head_logits.squeeze(2))
    return torch.stack(logits, dim=2), centers


def value_ensemble_stats(
    logits_all: torch.Tensor,
    centers_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head symexp expectations of (B,T,K,bins) logits -> ensemble mean and std, each (B,T).

    The std is the population std over heads (zero for one head), so ``mean - lambda * std`` is
    exactly the single-head value when K == 1.
    """
    probs = torch.softmax(logits_all.float(), dim=-1)
    expect_log = (probs * centers_log.to(probs.dtype)).sum(dim=-1)
    values = torch.sign(expect_log) * (torch.exp(expect_log.abs()) - 1.0)
    mean = values.mean(dim=2)
    std = values.std(dim=2, unbiased=False) if values.shape[2] > 1 else torch.zeros_like(mean)
    return mean, std


class RoachPpoFeatureEncoder(nn.Module):
    """Exact ROACH ``XtMaCNN`` feature extractor for 15x192x192 BEV masks.

    ROACH combines the image branch with its six-dimensional
    ``[control, vel_xy]`` vector and emits one 256-dimensional control feature.
    Keeping this module local makes chunk checkpoints self-contained and avoids
    importing Gym merely to reconstruct the PPO observation space.
    """

    birdview_shape = (15, 192, 192)
    state_dim = 6
    features_dim = 256

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(15, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.state_linear = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # A 192x192 input produces a 2x2x256 image feature (1024 values).
        self.linear = nn.Sequential(
            nn.Linear(1024 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def forward(self, birdview: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if birdview.ndim != 4 or tuple(birdview.shape[1:]) != self.birdview_shape:
            raise ValueError(
                "ROACH birdview must have shape (B,15,192,192), got "
                f"{tuple(birdview.shape)}"
            )
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"ROACH state must have shape (B,6), got {tuple(state.shape)}"
            )
        image_features = self.cnn(birdview.float() / 255.0)
        state_features = self.state_linear(state.float())
        return self.linear(torch.cat([image_features, state_features], dim=-1))


def load_roach_ppo_encoder_weights(
    encoder: RoachPpoFeatureEncoder,
    checkpoint: str | Path,
) -> None:
    """Strictly initialize a local encoder from a ROACH PPO checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ROACH PPO checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy_state = payload.get("policy_state_dict")
    if not isinstance(policy_state, Mapping):
        raise KeyError(f"{checkpoint} is missing policy_state_dict")
    prefix = "features_extractor."
    encoder_state = {
        key[len(prefix):]: value
        for key, value in policy_state.items()
        if key.startswith(prefix)
    }
    expected = encoder.state_dict()
    if set(encoder_state) != set(expected):
        missing = sorted(set(expected) - set(encoder_state))
        unexpected = sorted(set(encoder_state) - set(expected))
        raise RuntimeError(
            f"incompatible ROACH PPO feature extractor in {checkpoint}: "
            f"missing={missing} unexpected={unexpected}"
        )
    shape_mismatches = {
        key: (tuple(encoder_state[key].shape), tuple(expected[key].shape))
        for key in expected
        if tuple(encoder_state[key].shape) != tuple(expected[key].shape)
    }
    if shape_mismatches:
        raise RuntimeError(
            f"incompatible ROACH PPO encoder tensor shapes in {checkpoint}: "
            f"{shape_mismatches}"
        )
    encoder.load_state_dict(encoder_state, strict=True)


class MdVectorFeatureEncoder(nn.Module):
    """MLP for CAT's 101-dim MetaDrive observation, laid out as the TD3 actor's first two layers.

    ``l1``/``l2`` have the actor's names and shapes (cat/saferl_algo/TD3.py), so the expert's weights load
    directly; the output is the actor's 256-dim hidden feature after the second ReLU.
    """

    features_dim = 256

    def __init__(self, obs_dim: int = 101):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.l1 = nn.Linear(self.obs_dim, 256)
        self.l2 = nn.Linear(256, 256)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"MetaDrive observations must have shape (B,{self.obs_dim}), got {tuple(obs.shape)}")
        return F.relu(self.l2(F.relu(self.l1(obs.float()))))


def load_md_td3_encoder_weights(encoder: MdVectorFeatureEncoder, actor_file: str | Path) -> None:
    """Initialise ``encoder`` from a TD3 actor state dict file (``<prefix>_actor`` written by TD3.save)."""
    actor_file = Path(actor_file).expanduser().resolve()
    if not actor_file.is_file():
        raise FileNotFoundError(f"TD3 actor checkpoint not found: {actor_file}")
    state = torch.load(actor_file, map_location="cpu", weights_only=True)
    subset = {k: v for k, v in state.items() if k.split(".")[0] in ("l1", "l2")}
    encoder.load_state_dict(subset, strict=True)


class F1VectorFeatureEncoder(nn.Module):
    """MLP for the F1TENTH twin's 144-dim observation, laid out as the PPO actor's first two layers.

    ``l1``/``l2`` have the shapes of stable-baselines3's ``mlp_extractor.policy_net.0``/``.2``
    (mixed_quality_f1tenth/train_ppo.py: 144 -> 256 -> 256, tanh), so the expert's weights load directly; the
    output is the actor's 256-dim hidden feature after the second tanh.
    """

    features_dim = 256

    def __init__(self, obs_dim: int = 144):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.l1 = nn.Linear(self.obs_dim, 256)
        self.l2 = nn.Linear(256, 256)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"F1TENTH observations must have shape (B,{self.obs_dim}), got {tuple(obs.shape)}")
        return torch.tanh(self.l2(torch.tanh(self.l1(obs.float()))))


def load_f1_ppo_encoder_weights(encoder: F1VectorFeatureEncoder, sb3_checkpoint: str | Path) -> None:
    """Initialise ``encoder`` from the actor of a stable-baselines3 PPO checkpoint (zip) of train_ppo.py."""
    import io
    import zipfile

    path = Path(sb3_checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    with zipfile.ZipFile(path) as z:
        state = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    subset = {f"{name}.{kind}": state[f"mlp_extractor.policy_net.{i}.{kind}"]
              for name, i in (("l1", 0), ("l2", 2)) for kind in ("weight", "bias")}
    encoder.load_state_dict(subset, strict=True)


def canonicalize_model_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map saturated controls to the interior range representable by tanh policies."""
    return actions.clamp(-1.0 + MODEL_ACTION_EPS, 1.0 - MODEL_ACTION_EPS)


class _LatentTemporalScorer(nn.Module):
    """SimDist-style scorer trunk over explicit latent/action/task inputs."""

    def __init__(
        self,
        *,
        latent_dim: int,
        extra_dim: int,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(latent_dim) + int(extra_dim), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(round(float(mlp_ratio) * int(d_model))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, latents: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        if latents.shape[:2] != extras.shape[:2]:
            raise ValueError(
                f"scorer latent/extras time mismatch: {tuple(latents.shape)} vs {tuple(extras.shape)}"
            )
        x = torch.cat([latents.flatten(2), extras], dim=-1)
        x = self.input_proj(x)
        pos = sinusoid_table(x.shape[1], x.shape[2], device=x.device)
        x = x + pos[None].to(dtype=x.dtype) * (float(x.shape[2]) ** -0.5)
        return self.norm(self.temporal(x))


class ChunkWorldModel(nn.Module):
    """No-denoising H-step latent dynamics model.

    The dynamics follows SimDist's planning-oriented layout. Historical
    actions, states, and packed visual latents form an invariant history memory;
    the complete candidate action chunk forms the decoder queries. All future
    latents are predicted in one forward pass, while causal query attention
    prevents a prediction at step ``t`` from using later candidate actions.

    Args:
        action_dim: Number of controlled action dimensions trained/predicted by
            the policy head, e.g. 2 for CARLA [acc, steer].
        model_action_dim: Padded action width used by the dataset and
            ``ActionEncoder``. Existing CARLA shards use 16.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_spatial: int,
        n_spatial: int,
        horizon: int,
        action_dim: int = 2,
        model_action_dim: int = 16,
        n_register: int = 4,
        n_agent: int = 1,
        n_heads: int = 4,
        depth: int = 6,
        policy_depth: int = 3,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_cautious",
        n_tasks: int = 128,
        reward_num_bins: int = 101,
        reward_log_low: float = -3.0,
        reward_log_high: float = 3.0,
        value_num_bins: int = 101,
        value_log_low: float = -10.0,
        value_log_high: float = 10.0,
        policy_log_std_init: float = -2.0,
        use_states: bool = False,
        state_dim: int = 6,
        use_agent_heads: bool = True,
        scorer_arch: str = SCORER_ARCH_SIMDIST,
        dynamics_arch: str = DYNAMICS_ARCH_SIMDIST,
        latent_codec_arch: str = LATENT_CODEC_MLP,
        observation_representation: str = OBSERVATION_REPRESENTATION_TOKENIZER,
        value_num_heads: int = 1,
        value_head_dropout: float = 0.0,
        scene_factor_dim: int = 0,
        reward_factor_kind: str = REWARD_FACTOR_KIND_VALEO,
        md_obs_dim: int = 101,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.model_action_dim = int(model_action_dim)
        self.n_register = int(n_register)
        self.n_agent = int(n_agent)
        self.use_states = bool(use_states)
        self.state_dim = int(state_dim)
        self.use_agent_heads = bool(use_agent_heads)
        # Value ensemble (TD-MPC2 style): K independently initialised heads on the shared value trunk.
        # K = 1 is the original single head; the planner can score with mean - lambda * std.
        self.value_num_heads = max(1, int(value_num_heads))
        # Scene-factor auxiliary head: rule-derived scene state (light state, lead, pedestrian, ...)
        # predicted from the keyframe and every imagined future latent; 0 disables the head.
        self.scene_factor_dim = max(0, int(scene_factor_dim))
        self.scorer_arch = str(scorer_arch)
        self.dynamics_arch = str(dynamics_arch)
        self.latent_codec_arch = str(latent_codec_arch)
        self.observation_representation = str(observation_representation)
        if self.observation_representation not in (
            OBSERVATION_REPRESENTATION_TOKENIZER,
            OBSERVATION_REPRESENTATION_ROACH_PPO,
            OBSERVATION_REPRESENTATION_MD_VECTOR,
            OBSERVATION_REPRESENTATION_F1_VECTOR,
        ):
            raise ValueError(
                "unsupported observation_representation="
                f"{self.observation_representation!r}"
            )
        self.reward_factor_kind = str(reward_factor_kind)
        if self.reward_factor_kind not in (REWARD_FACTOR_KIND_VALEO, REWARD_FACTOR_KIND_CAT, REWARD_FACTOR_KIND_F1TENTH):
            raise ValueError(f"unsupported reward_factor_kind={self.reward_factor_kind!r}")
        self.md_obs_dim = int(md_obs_dim)
        if self.scorer_arch not in (SCORER_ARCH_LEGACY, SCORER_ARCH_SIMDIST):
            raise ValueError(
                f"unsupported scorer_arch={self.scorer_arch!r}; expected "
                f"{SCORER_ARCH_LEGACY!r} or {SCORER_ARCH_SIMDIST!r}"
            )
        if self.dynamics_arch != DYNAMICS_ARCH_SIMDIST:
            raise ValueError(
                f"unsupported dynamics_arch={self.dynamics_arch!r}; expected "
                f"{DYNAMICS_ARCH_SIMDIST!r}"
            )
        if self.latent_codec_arch != LATENT_CODEC_MLP:
            raise ValueError(
                f"unsupported latent_codec_arch={self.latent_codec_arch!r}; "
                f"expected {LATENT_CODEC_MLP!r}"
            )

        if self.n_spatial != 1:
            raise ValueError(
                "simdist_action_decoder requires one packed visual token with "
                "a learned latent codec; got "
                f"n_spatial={self.n_spatial}, d_spatial={self.d_spatial}, "
                f"d_model={self.d_model}. Re-cache tokenizer latents with "
                "packing_factor=n_latents."
            )
        if self.d_spatial <= 0 or self.d_model <= 0:
            raise ValueError("d_spatial and d_model must be positive")
        if self.observation_representation == OBSERVATION_REPRESENTATION_ROACH_PPO:
            if (self.n_spatial, self.d_spatial, self.d_model) != (1, 256, 256):
                raise ValueError(
                    "roach_ppo representation requires n_spatial=1, "
                    "d_spatial=d_model=256; got "
                    f"({self.n_spatial},{self.d_spatial},{self.d_model})"
                )
            if self.use_states:
                raise ValueError(
                    "roach_ppo features already fuse the exact six-dimensional "
                    "PPO state; set use_states=False to avoid conditioning twice"
                )
            self.observation_encoder = RoachPpoFeatureEncoder()
        elif self.observation_representation == OBSERVATION_REPRESENTATION_MD_VECTOR:
            if (self.n_spatial, self.d_spatial, self.d_model) != (1, 256, 256):
                raise ValueError(
                    "md_vector representation requires n_spatial=1, d_spatial=d_model=256; got "
                    f"({self.n_spatial},{self.d_spatial},{self.d_model})"
                )
            if self.use_states:
                raise ValueError("md_vector observations already contain the ego state; set use_states=False")
            self.observation_encoder = MdVectorFeatureEncoder(self.md_obs_dim)
        elif self.observation_representation == OBSERVATION_REPRESENTATION_F1_VECTOR:
            if (self.n_spatial, self.d_spatial, self.d_model) != (1, 256, 256):
                raise ValueError(
                    "f1_vector representation requires n_spatial=1, d_spatial=d_model=256; got "
                    f"({self.n_spatial},{self.d_spatial},{self.d_model})"
                )
            if self.use_states:
                raise ValueError("f1_vector observations already contain the ego state; set use_states=False")
            self.observation_encoder = F1VectorFeatureEncoder(self.md_obs_dim)
        if self.d_spatial == self.d_model:
            # Preserve exact checkpoint compatibility with the earlier compact
            # tokenizer setup: Identity contributes no state-dict entries.
            self.latent_compressor = nn.Identity()
            self.latent_decompressor = nn.Identity()
        else:
            # The tokenizer latent remains the reconstruction target, while all
            # planning-time dynamics and scorers operate in d_model dimensions.
            self.latent_compressor = nn.Sequential(
                nn.LayerNorm(self.d_spatial),
                nn.Linear(self.d_spatial, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self.latent_decompressor = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_spatial),
            )
        self.memory_type_embed = nn.Parameter(torch.empty(3, self.d_model))
        nn.init.normal_(self.memory_type_embed, std=0.02)
        self.policy_queries = nn.Parameter(torch.empty(self.horizon, self.d_model))
        nn.init.normal_(self.policy_queries, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=self.model_action_dim)
        if self.use_states:
            self.state_encoder = StateEncoder(d_model=self.d_model, state_dim=self.state_dim)
            if self.use_agent_heads:
                self.state_head = StatePredictorHead(d_model=self.d_model, state_dim=self.state_dim)
        self.task_embedder = TaskEmbedder(
            d_model=self.d_model,
            n_agent=self.n_agent,
            use_ids=True,
            n_tasks=int(n_tasks),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(round(float(mlp_ratio) * self.d_model)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(depth),
            norm=nn.LayerNorm(self.d_model),
        )
        self.latent_head = nn.Identity()
        causal_mask = torch.ones(self.horizon, self.horizon, dtype=torch.bool).triu(1)
        self.register_buffer("future_causal_mask", causal_mask, persistent=False)
        if self.use_agent_heads:
            policy_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=int(n_heads),
                dim_feedforward=int(round(float(mlp_ratio) * self.d_model)),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.policy_transformer = nn.TransformerDecoder(
                policy_layer,
                num_layers=int(policy_depth),
                norm=nn.LayerNorm(self.d_model),
            )
            self.reward_head = RewardHeadMTP(
                d_model=self.d_model,
                L=1,
                num_bins=int(reward_num_bins),
                log_low=float(reward_log_low),
                log_high=float(reward_log_high),
                mlp_ratio=2.0,
                dropout=0.0,
            )
            # The five nonnegative factors are [ego speed, desired speed,
            # lateral distance, heading error, |terminal reward|].
            self.reward_factor_head = VectorHeadMTP(
                d_model=self.d_model,
                output_dim=5,
                L=1,
                mlp_ratio=2.0,
                dropout=0.0,
            )
            self.safety_event_head = VectorHeadMTP(
                d_model=self.d_model,
                output_dim=3,
                L=1,
                mlp_ratio=2.0,
                dropout=0.0,
            )
            self.continuation_head = ContinuationHeadMTP(
                d_model=self.d_model,
                L=1,
                mlp_ratio=2.0,
                dropout=0.0,
            )
            self.value_heads = nn.ModuleList(
                [
                    ValueHeadMTP(
                        d_model=self.d_model,
                        L=1,
                        num_bins=int(value_num_bins),
                        log_low=float(value_log_low),
                        log_high=float(value_log_high),
                        mlp_ratio=2.0,
                        dropout=float(value_head_dropout),
                    )
                    for _ in range(self.value_num_heads)
                ]
            )
            if self.scene_factor_dim > 0:
                self.scene_factor_head = nn.Sequential(
                    nn.LayerNorm(self.n_spatial * self.d_model),
                    nn.Linear(self.n_spatial * self.d_model, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.scene_factor_dim),
                )
            else:
                self.scene_factor_head = None
            self.policy_head = PolicyHeadMTP(
                d_model=self.d_model,
                action_dim=self.action_dim,
                L=1,
                kind="continuous",
                mlp_ratio=2.0,
                dropout=0.0,
            )
            self.policy_logstd_head = PolicyHeadMTP(
                d_model=self.d_model,
                action_dim=self.action_dim,
                L=1,
                kind="continuous",
                mlp_ratio=2.0,
                dropout=0.0,
            )
            with torch.no_grad():
                nn.init.zeros_(self.policy_head.out.weight)
                nn.init.zeros_(self.policy_head.out.bias)
                nn.init.zeros_(self.policy_logstd_head.out.weight)
                self.policy_logstd_head.out.bias.fill_(float(policy_log_std_init))

            if self.scorer_arch == SCORER_ARCH_SIMDIST:
                planning_latent_dim = self.n_spatial * self.d_model
                # SimDist rewards are conditioned on the state before the
                # transition, the candidate action, and the command/task.
                self.reward_scorer = _LatentTemporalScorer(
                    latent_dim=planning_latent_dim,
                    extra_dim=self.action_dim + self.d_model
                    + (self.state_dim if self.use_states else 0),
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    mlp_ratio=float(mlp_ratio),
                    depth=1,
                    dropout=float(dropout),
                )
                # Value and post-transition heads consume predicted latents.
                self.value_scorer = _LatentTemporalScorer(
                    latent_dim=planning_latent_dim,
                    extra_dim=self.d_model
                    + (self.state_dim if self.use_states else 0),
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    mlp_ratio=float(mlp_ratio),
                    depth=1,
                    dropout=float(dropout),
                )
                if self.use_states:
                    self.state_scorer = _LatentTemporalScorer(
                        latent_dim=planning_latent_dim,
                        extra_dim=self.d_model,
                        d_model=self.d_model,
                        n_heads=int(n_heads),
                        mlp_ratio=float(mlp_ratio),
                        depth=1,
                        dropout=float(dropout),
                    )

    @property
    def value_head(self) -> nn.Module:
        """First value head; kept for callers written against the single-head layout."""
        return self.value_heads[0]

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        # Single-head checkpoints saved ``value_head.*``; the ensemble stores ``value_heads.<k>.*``.
        return super().load_state_dict(remap_legacy_value_head_keys(state_dict), strict=strict, **kwargs)

    def _resolve_value_heads(
        self, scorer_modules: Optional[Mapping[str, nn.Module]]
    ) -> nn.Module | list[nn.Module]:
        if scorer_modules is not None:
            if "value_heads" in scorer_modules:
                return scorer_modules["value_heads"]
            if "value_head" in scorer_modules:
                return scorer_modules["value_head"]
        return self.value_heads

    def scene_factor_logits(
        self,
        planning_context: torch.Tensor,
        planning_z_pred: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """(B, H+1, F) scene-factor logits for the keyframe (index 0) and the H imagined latents."""
        if self.scene_factor_head is None:
            return None
        latents = torch.cat([planning_context[:, -1:], planning_z_pred], dim=1)
        return self.scene_factor_head(latents.flatten(2).float())

    def _task_features(
        self,
        task_ids: torch.Tensor,
        *,
        B: int,
        T: int,
    ) -> torch.Tensor:
        if self.n_agent <= 0:
            return torch.zeros(
                B,
                T,
                self.d_model,
                device=task_ids.device,
                dtype=self.memory_type_embed.dtype,
            )
        return self.task_embedder(task_ids, B=B, T=T).mean(dim=2)

    def _reward_factor_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """ValeoAction factors are nonnegative (softplus); CAT components and F1TENTH factors stay raw (md_reward.py
        and f1_reward.py read them)."""
        if self.reward_factor_kind in (REWARD_FACTOR_KIND_CAT, REWARD_FACTOR_KIND_F1TENTH):
            return raw
        return F.softplus(raw)

    def _add_temporal_positions(self, x: torch.Tensor) -> torch.Tensor:
        pos = sinusoid_table(x.shape[1], x.shape[2], device=x.device)
        return x + pos[None].to(dtype=x.dtype) * (float(x.shape[2]) ** -0.5)

    def compress_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Map packed tokenizer latents to the compact planning space."""
        if latents.ndim != 4:
            raise ValueError(
                f"latents must have shape (B,T,N,D), got {tuple(latents.shape)}"
            )
        if latents.shape[-2:] != (self.n_spatial, self.d_spatial):
            raise ValueError(
                f"raw latent shape {tuple(latents.shape[-2:])} != "
                f"({self.n_spatial}, {self.d_spatial})"
            )
        return self.latent_compressor(latents)

    def encode_ppo_observations(
        self,
        birdview: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode exact ROACH observations to ``(B,T,1,256)`` planning states."""
        if self.observation_representation != OBSERVATION_REPRESENTATION_ROACH_PPO:
            raise RuntimeError(
                "encode_ppo_observations requires observation_representation=roach_ppo"
            )
        if birdview.ndim != 5 or tuple(birdview.shape[2:]) != (15, 192, 192):
            raise ValueError(
                "ROACH sequence birdview must have shape (B,T,15,192,192), got "
                f"{tuple(birdview.shape)}"
            )
        if state.ndim != 3 or state.shape[:2] != birdview.shape[:2] or state.shape[-1] != 6:
            raise ValueError(
                "ROACH sequence state must have shape (B,T,6) aligned with birdview, got "
                f"{tuple(state.shape)}"
            )
        B, T = birdview.shape[:2]
        features = self.observation_encoder(
            birdview.reshape(B * T, 15, 192, 192),
            state.reshape(B * T, 6),
        )
        return features.view(B, T, 1, 256)

    def encode_md_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode vector observation sequences (B,T,obs_dim) to ``(B,T,1,256)`` planning states (MetaDrive's
        md_vector, F1TENTH's f1_vector)."""
        if self.observation_representation not in VECTOR_REPRESENTATIONS:
            raise RuntimeError("encode_md_observations requires observation_representation md_vector or f1_vector")
        if obs.ndim != 3 or obs.shape[-1] != self.md_obs_dim:
            raise ValueError(
                f"MetaDrive observation sequence must have shape (B,T,{self.md_obs_dim}), got {tuple(obs.shape)}"
            )
        B, T = obs.shape[:2]
        return self.observation_encoder(obs.reshape(B * T, self.md_obs_dim)).view(B, T, 1, 256)

    def decompress_latents(self, planning_latents: torch.Tensor) -> torch.Tensor:
        """Map compact planning latents back to tokenizer latent coordinates."""
        if planning_latents.ndim != 4:
            raise ValueError(
                "planning_latents must have shape (B,T,N,D), got "
                f"{tuple(planning_latents.shape)}"
            )
        if planning_latents.shape[-2:] != (self.n_spatial, self.d_model):
            raise ValueError(
                f"planning latent shape {tuple(planning_latents.shape[-2:])} != "
                f"({self.n_spatial}, {self.d_model})"
            )
        return self.latent_decompressor(planning_latents)

    def _build_history_memory(
        self,
        z_context: torch.Tensor,
        actions_context: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        state_context: Optional[torch.Tensor] = None,
        planning_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode candidate-invariant history once as decoder memory."""
        B, C, N, D = z_context.shape
        if N != self.n_spatial or D != self.d_spatial:
            raise ValueError(
                f"z_context has shape {tuple(z_context.shape)}, expected "
                f"(B,C,{self.n_spatial},{self.d_spatial})"
            )
        if actions_context.shape[:2] != (B, C):
            raise ValueError("actions_context must align with z_context time dimension")
        if actions_context.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"actions last dim {actions_context.shape[-1]} != "
                f"model_action_dim={self.model_action_dim}"
            )
        action_tokens = self.action_encoder(
            canonicalize_model_actions(actions_context),
            batch_time_shape=(B, C),
            act_mask=None,
            as_tokens=False,
        ) + self.memory_type_embed[0].view(1, 1, -1)
        history_positions = sinusoid_table(
            C, self.d_model, device=z_context.device
        ).to(dtype=action_tokens.dtype)
        history_positions = history_positions * (float(self.d_model) ** -0.5)
        action_tokens = action_tokens + history_positions[:C].view(1, C, -1)

        if planning_context is None:
            planning_context = self.compress_latents(z_context)
        if planning_context.shape != (B, C, self.n_spatial, self.d_model):
            raise ValueError(
                "planning_context shape mismatch: got "
                f"{tuple(planning_context.shape)}, expected "
                f"({B}, {C}, {self.n_spatial}, {self.d_model})"
            )
        # Keep each compact BEV frame as one decoder-memory token.
        visual_tokens = planning_context.squeeze(2)
        visual_tokens = visual_tokens + self.memory_type_embed[2].view(1, 1, -1)
        visual_tokens = visual_tokens + self._task_features(
            task_ids, B=B, T=C
        ).to(dtype=visual_tokens.dtype)
        visual_tokens = visual_tokens + history_positions[:C].view(1, C, -1)

        if self.use_states:
            if state_context is None:
                state_tokens = self.state_encoder(
                    None, batch_time_shape=(B, C)
                ).squeeze(2)
            else:
                if state_context.shape[:2] != (B, C):
                    raise ValueError("state_context must align with z_context time dimension")
                if state_context.shape[-1] < self.state_dim:
                    raise ValueError(
                        f"state_context last dim {state_context.shape[-1]} < state_dim={self.state_dim}"
                    )
                state_tokens = self.state_encoder(
                    normalize_state_vec(
                        state_context[..., : self.state_dim].float()
                    ),
                    batch_time_shape=(B, C),
                ).squeeze(2)
            state_tokens = state_tokens + self.memory_type_embed[1].view(1, 1, -1)
            state_tokens = state_tokens + history_positions[:C].view(1, C, -1)
            memory = torch.stack(
                [state_tokens, action_tokens, visual_tokens], dim=2
            ).flatten(1, 2)
        else:
            memory = torch.stack(
                [action_tokens, visual_tokens], dim=2
            ).flatten(1, 2)
        return memory

    def encode_context(
        self,
        z_context: torch.Tensor,
        actions_context: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        state_context: Optional[torch.Tensor] = None,
        planning_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return candidate-invariant decoder memory for policy and planning."""
        return self._build_history_memory(
            z_context,
            actions_context,
            task_ids,
            state_context=state_context,
            planning_context=planning_context,
        )

    def _build_action_queries(
        self,
        actions_future: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, H = actions_future.shape[:2]
        if H > self.horizon:
            raise ValueError(f"future horizon {H} exceeds model horizon {self.horizon}")
        if actions_future.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"future action dim {actions_future.shape[-1]} != "
                f"model_action_dim={self.model_action_dim}"
            )
        queries = self.action_encoder(
            canonicalize_model_actions(actions_future),
            batch_time_shape=(B, H),
            act_mask=None,
            as_tokens=False,
        )
        queries = queries + self._task_features(
            task_ids, B=B, T=H
        ).to(dtype=queries.dtype)
        return self._add_temporal_positions(queries)

    def _build_policy_queries(
        self,
        task_ids: torch.Tensor,
        *,
        B: int,
        H: int,
    ) -> torch.Tensor:
        queries = self.policy_queries[:H].view(1, H, -1).expand(B, -1, -1)
        queries = queries + self._task_features(
            task_ids, B=B, T=H
        ).to(dtype=queries.dtype)
        return self._add_temporal_positions(queries)

    def _decode_queries(
        self,
        decoder: nn.Module,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        H = int(queries.shape[1])
        return decoder(
            queries,
            memory,
            tgt_mask=self.future_causal_mask[:H, :H],
            tgt_is_causal=True,
        )

    def predict_policy(
        self,
        z_context: torch.Tensor,
        actions_context: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        state_context: Optional[torch.Tensor] = None,
        policy_modules: Optional[Mapping[str, nn.Module]] = None,
        context_memory: Optional[torch.Tensor] = None,
        planning_context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict the BC action chunk without running dynamics/scorer heads."""
        policy_transformer = (
            policy_modules["policy_transformer"]
            if policy_modules is not None
            else self.policy_transformer
        )
        policy_head = (
            policy_modules["policy_head"]
            if policy_modules is not None
            else self.policy_head
        )
        policy_logstd_head = (
            policy_modules["policy_logstd_head"]
            if policy_modules is not None
            else self.policy_logstd_head
        )
        B = int(z_context.shape[0])
        H = self.horizon
        memory = context_memory
        if memory is None:
            memory = self.encode_context(
                z_context,
                actions_context,
                task_ids,
                state_context=state_context,
                planning_context=planning_context,
            )
        elif int(memory.shape[0]) != B:
            raise ValueError(
                f"context_memory batch {int(memory.shape[0])} != policy batch {B}"
            )
        pol_queries = self._build_policy_queries(task_ids, B=B, H=H)
        # The policy transformer runs in fp32 even under autocast. Its first self-attention layer learns
        # a near-hard attention over the learned queries: run 2ulpkjq8 reached |q|,|k| ~ 5e3 and logits
        # ~ 2e8 (fp16 max 6.5e4) by step 360k, and from step 367k the fp16 attention backward returned
        # NaN on every batch, silently stopping training. The module is small (3 layers, 16 queries),
        # so the fp32 cost is negligible; the dynamics decoder stays under autocast (logits ~ 3e2).
        with torch.autocast(device_type=pol_queries.device.type, enabled=False):
            pol_h = self._decode_queries(
                policy_transformer, pol_queries.float(), memory.float()
            )
        return {
            "policy_mean_pre": policy_head(pol_h).squeeze(2),
            "policy_logstd": policy_logstd_head(pol_h).squeeze(2),
        }

    def forward(
        self,
        z_context: Optional[torch.Tensor],
        actions_context: torch.Tensor,
        actions_future: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        compute_policy: bool = True,
        compute_agent_heads: bool = True,
        detach_scorer_inputs: bool = False,
        scorer_modules: Optional[Mapping[str, nn.Module]] = None,
        state_context: Optional[torch.Tensor] = None,
        context_memory: Optional[torch.Tensor] = None,
        planning_context: Optional[torch.Tensor] = None,
        decode_latents: bool = True,
        ppo_observations: Optional[torch.Tensor] = None,
        ppo_states: Optional[torch.Tensor] = None,
        md_observations: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict a chunk of future latents and scalar heads."""
        B = int(actions_future.shape[0])
        H = int(actions_future.shape[1])
        encoder_latent_targets = None
        if md_observations is not None:
            if ppo_observations is not None or ppo_states is not None:
                raise ValueError("md_observations cannot be combined with ROACH PPO observations")
            if self.observation_representation not in VECTOR_REPRESENTATIONS:
                raise ValueError("md_observations require observation_representation md_vector or f1_vector")
            if context_memory is not None or planning_context is not None:
                raise ValueError("raw MetaDrive observations cannot be combined with cached context")
            context_steps = int(actions_context.shape[1])
            if int(md_observations.shape[1]) != context_steps + H:
                raise ValueError(
                    "MetaDrive observation sequence length must equal context+horizon: "
                    f"got {int(md_observations.shape[1])}, expected {context_steps + H}"
                )
            encoded = self.encode_md_observations(md_observations)
            z_context = encoded[:, :context_steps].contiguous()
            # as for ROACH PPO (SimDist): future encoder targets carry no gradient
            encoder_latent_targets = encoded[:, context_steps:context_steps + H].detach().contiguous()
        elif ppo_observations is not None or ppo_states is not None:
            if ppo_observations is None or ppo_states is None:
                raise ValueError(
                    "ppo_observations and ppo_states must be provided together"
                )
            if self.observation_representation != OBSERVATION_REPRESENTATION_ROACH_PPO:
                raise ValueError(
                    "raw PPO observations require observation_representation=roach_ppo"
                )
            if context_memory is not None or planning_context is not None:
                raise ValueError(
                    "raw PPO observations cannot be combined with cached context"
                )
            context_steps = int(actions_context.shape[1])
            expected_steps = context_steps + H
            if int(ppo_observations.shape[1]) != expected_steps:
                raise ValueError(
                    "raw PPO observation sequence length must equal context+horizon: "
                    f"got {int(ppo_observations.shape[1])}, expected {expected_steps}"
                )
            encoded = self.encode_ppo_observations(
                ppo_observations,
                ppo_states,
            )
            z_context = encoded[:, :context_steps].contiguous()
            # SimDist stops gradients through future encoder targets while the
            # observed context remains differentiable through the world loss.
            encoder_latent_targets = encoded[
                :, context_steps:context_steps + H
            ].detach().contiguous()
        elif z_context is None:
            raise ValueError(
                "z_context is required unless raw PPO observations are provided"
            )
        if planning_context is None:
            planning_context = self.compress_latents(z_context)
        expected_planning_shape = (
            B,
            int(actions_context.shape[1]),
            self.n_spatial,
            self.d_model,
        )
        if planning_context.shape != expected_planning_shape:
            raise ValueError(
                "planning_context shape mismatch: got "
                f"{tuple(planning_context.shape)}, expected {expected_planning_shape}"
            )
        memory = context_memory
        if memory is None:
            memory = self.encode_context(
                z_context,
                actions_context,
                task_ids,
                state_context=state_context,
                planning_context=planning_context,
            )
        elif int(memory.shape[0]) != B:
            raise ValueError(
                f"context_memory batch {int(memory.shape[0])} != dynamics batch {B}"
            )
        action_queries = self._build_action_queries(actions_future, task_ids)
        fut_h = self._decode_queries(self.dynamics, action_queries, memory)
        planning_z_pred = self.latent_head(fut_h).view(
            B, H, self.n_spatial, self.d_model
        )
        out = {
            "planning_latents": planning_z_pred,
            "h": fut_h,
            # Encoded context and its memory, so a caller that needs a second rollout under a
            # different action sequence (e.g. the risk head's braking-monotonicity term) can
            # reuse them instead of paying for the observation encoder again.
            "z_context": z_context,
            "context_memory": memory,
            # observed context in planning coordinates; the roach_ppo path encodes it here, and returning it
            # lets auxiliary heads (e.g. the CoC language head) read the keyframe without re-running the CNN.
            "planning_context": planning_context,
        }
        if encoder_latent_targets is not None:
            out["encoder_latent_targets"] = encoder_latent_targets
        if decode_latents:
            out["latents"] = self.decompress_latents(planning_z_pred)
        if not self.use_agent_heads or not compute_agent_heads:
            return out

        reward_head = (
            scorer_modules["reward_head"]
            if scorer_modules is not None and "reward_head" in scorer_modules
            else self.reward_head
        )
        reward_factor_head = (
            scorer_modules["reward_factor_head"]
            if scorer_modules is not None and "reward_factor_head" in scorer_modules
            else self.reward_factor_head
        )
        continuation_head = (
            scorer_modules["continuation_head"]
            if scorer_modules is not None and "continuation_head" in scorer_modules
            else self.continuation_head
        )
        value_heads = self._resolve_value_heads(scorer_modules)
        if self.scorer_arch == SCORER_ARCH_SIMDIST:
            reward_scorer = (
                scorer_modules["reward_scorer"]
                if scorer_modules is not None
                else self.reward_scorer
            )
            # For reward r_t, use observed z_t at the first slot and predicted
            # z_{t+j} for later slots, matching SimDist's z_t_tm1 construction.
            task_features = self._task_features(task_ids, B=B, T=H)
            if self.use_states:
                state_features = self.state_scorer(
                    planning_z_pred.float(),
                    task_features.float(),
                )
                state_pred = self.state_head(state_features)
                if state_context is None:
                    current_state = torch.zeros(
                        B,
                        1,
                        self.state_dim,
                        device=z_context.device,
                        dtype=z_context.dtype,
                    )
                else:
                    current_state = normalize_state_vec(
                        state_context[:, -1:, : self.state_dim].float()
                    )
            else:
                state_features = None
                state_pred = None
            scorer_z_pred = (
                planning_z_pred.detach()
                if detach_scorer_inputs else planning_z_pred
            )
            scorer_state_pred = (
                state_pred.detach()
                if detach_scorer_inputs and state_pred is not None
                else state_pred
            )
            z_before = torch.cat(
                [planning_context[:, -1:], scorer_z_pred[:, :-1]],
                dim=1,
            )
            if self.use_states:
                state_before = torch.cat(
                    [current_state, scorer_state_pred[:, :-1]],
                    dim=1,
                )
                reward_state_extra = state_before
                value_state_extra = scorer_state_pred
            else:
                reward_state_extra = planning_z_pred.new_zeros(B, H, 0)
                value_state_extra = planning_z_pred.new_zeros(B, H, 0)
            reward_actions = canonicalize_model_actions(actions_future)[
                :, :, : self.action_dim
            ].float()
            reward_features = reward_scorer(
                z_before.float(),
                torch.cat(
                    [
                        reward_actions,
                        reward_state_extra.float(),
                        task_features.float(),
                    ],
                    dim=-1,
                ),
            )
            value_features = self.value_scorer(
                scorer_z_pred.float(),
                torch.cat(
                    [value_state_extra.float(), task_features.float()],
                    dim=-1,
                ),
            )
            reward_logits, reward_centers = reward_head(reward_features)
            reward_factors = self._reward_factor_activation(
                reward_factor_head(reward_features).squeeze(2)
            )
            cont_logits = continuation_head(value_features)
            value_logits_all, value_centers = value_ensemble_logits(value_heads, value_features)
        else:
            if scorer_modules is not None and "reward_scorer" in scorer_modules:
                raise ValueError(
                    "reward_scorer override is only valid for simdist_latent scorers"
                )
            scorer_h = fut_h.detach() if detach_scorer_inputs else fut_h
            reward_logits, reward_centers = reward_head(scorer_h)
            reward_factors = self._reward_factor_activation(
                reward_factor_head(scorer_h).squeeze(2)
            )
            cont_logits = continuation_head(scorer_h)
            value_logits_all, value_centers = value_ensemble_logits(value_heads, scorer_h)
            state_features = fut_h
            value_features = scorer_h
        out.update(
            {
                "reward_logits": reward_logits.squeeze(2),
                "reward_factors": reward_factors,
                "reward_centers": reward_centers,
                "cont_logits": cont_logits.squeeze(2),
                # head 0 keeps the single-head contract; the ensemble is (B,H,K,bins)
                "value_logits": value_logits_all[:, :, 0],
                "value_logits_all": value_logits_all,
                "value_centers": value_centers,
                "value_features": value_features,
            }
        )
        scene_logits = self.scene_factor_logits(planning_context, planning_z_pred)
        if scene_logits is not None:
            out["scene_logits"] = scene_logits
        if self.use_states:
            out["state_pred"] = (
                state_pred
                if self.scorer_arch == SCORER_ARCH_SIMDIST
                else self.state_head(state_features)
            )
        if compute_policy:
            out.update(self.predict_policy(
                z_context,
                actions_context,
                task_ids,
                state_context=state_context,
                context_memory=memory,
                planning_context=planning_context,
            ))
        return out

    def build_value_features(
        self,
        latents: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        normalized_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build value features for an arbitrary sequence of latent states.

        This is used by online Dreamer-style value learning to score
        ``[s_0, ..., s_H]`` with one shared temporal trunk. State inputs, when
        enabled, must already use the model's normalized state convention.
        """
        if self.scorer_arch != SCORER_ARCH_SIMDIST:
            raise RuntimeError(
                "online value features require scorer_arch=simdist_latent"
            )
        B, T = latents.shape[:2]
        task_features = self._task_features(task_ids, B=B, T=T)
        if self.use_states:
            if normalized_states is None:
                raise ValueError("normalized_states are required when use_states=True")
            if normalized_states.shape[:2] != (B, T):
                raise ValueError(
                    "value state sequence mismatch: "
                    f"{tuple(normalized_states.shape)} vs latents {tuple(latents.shape)}"
                )
            state_extra = normalized_states[..., : self.state_dim]
        else:
            state_extra = latents.new_zeros(B, T, 0)
        return self.value_scorer(
            latents.float(),
            torch.cat([state_extra.float(), task_features.float()], dim=-1),
        )
