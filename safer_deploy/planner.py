"""SAFER's controllers: MPPI over the world model with the learned risk term ('safer'), and the world model's BC prior.

This is the planner that was evaluated in simulation, extracted from the training code (dreamer4/train_online_loop.py:
the eager path of _mppi_plan_actions, _mppi_score_candidates, _sample_mppi_action_candidates,
_chunk_policy_mean_logstd and their helpers; mixed_quality_f1tenth/planner.py for loading). The statements are the
training code's; the branches that are off in SAFER's F1TENTH configuration (hazard terms, CoC risk, safe-brake and
elite-memory candidates, coloured noise) are left out, and MPPI_CONFIG, the configuration they ran with, is fixed here.
Given the same random state, SaferPlanner.act returns the training code's actions bit for bit (checked on recorded
Gazebo states).

Per decision: the observation is encoded, the context is the last 16 latents and executed actions (the first
observation repeated and zero actions before the start, as in training), the BC prior proposes a 16-step chunk, and two
MPPI iterations score 135 chunks each (128 samples around the current plan, 6 around the prior, the prior itself) by
the world model's predicted rewards, continuation and terminal value minus W times the risk head's penalty; the first
action of the final plan is executed and the plan, shifted by one step, starts the next decision.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .wm import f1_reward
from .wm.chunk_world_model import (
    OBSERVATION_REPRESENTATION_F1_VECTOR,
    REWARD_FACTOR_KIND_F1TENTH,
    ChunkWorldModel,
    value_ensemble_logits,
    value_ensemble_stats,
)
from .wm.risk_head import F1_EVENTS, RiskHead

# mixed_quality_f1tenth/planner.py: CARLA's online settings (train_online_loop._effective_mppi_cfg of them)
MPPI_CONFIG = dict(k=128, iterations=2, num_elites=16, mixture_coef=0.05, temperature=0.25, momentum=0.0,
                   init_acc_std=0.25, init_steer_std=0.08, min_acc_std=0.05, min_steer_std=0.02,
                   base_acc_std=0.10, base_steer_std=0.04, warm_start=True, gamma=0.99, horizon=16, context_length=16,
                   action_dim=2, value_score_weight=1.0)


# ------------------------------------------------------------------ helpers (train_policy / train_online_loop)
def _symexp(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


def symlog_mean_from_logits(logits: torch.Tensor, centers_log: torch.Tensor) -> torch.Tensor:
    """Expectation of symlog value from categorical logits. (B,T) or (B,T,K)."""
    if logits.dim() == 2:
        return logits
    probs = F.softmax(logits, dim=-1)
    if probs.dim() == 4:
        probs = probs[:, :, 0, :]
    return (probs * centers_log.view(1, 1, -1)).sum(dim=-1)


def _decode_reward_distribution(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return _symexp(symlog_mean_from_logits(logits.float(), centers))


def _blended_predicted_rewards(out: dict, actions: torch.Tensor, previous_steer: torch.Tensor) -> torch.Tensor:
    """The fixed 50/50 blend of the reward head's scalar and the F1TENTH reward rebuilt from the reward factors."""
    scalar_reward = _decode_reward_distribution(out["reward_logits"], out["reward_centers"])
    steer = actions[..., 1].float()
    steer_changes = (steer - torch.cat([previous_steer.float(), steer[:, :-1]], dim=1)).abs()
    decomposed_reward = f1_reward.expected_reward(out["reward_factors"], steer_changes)
    return 0.5 * scalar_reward + 0.5 * decomposed_reward


def _bound_mppi_action_sequences(actions: torch.Tensor) -> torch.Tensor:
    return actions.clamp(-0.999, 0.999)


def left_padded_context(latents, actions, ctx: int, action_dim: int):
    """The planner's context from the episode so far (md_dataset.left_padded_context): the first observation repeated
    before the start, a zero action at every slot without a preceding executed action."""
    history = list(latents)[-ctx:]
    n = len(history)
    observed = torch.cat(history, dim=1)
    if n < ctx:
        observed = torch.cat([observed[:, :1].expand(-1, ctx - n, -1, -1), observed], dim=1)
    recent = list(actions)[-(n - 1):] if n > 1 else []
    zeros = [torch.zeros(action_dim, dtype=observed.dtype, device=observed.device) for _ in range(ctx - len(recent))]
    ctx_act = torch.stack([*zeros, *[a.to(observed.dtype).to(observed.device) for a in recent]], dim=0)
    return observed, ctx_act[None]


# ------------------------------------------------------------------ loading
def load_checkpoint(path: str, device: torch.device) -> dict:
    """World model, frozen BC prior, scorers, value ensemble and risk head of a SAFER checkpoint (offline or fine-tuned:
    the same format)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    if cfg.get("observation_representation") != OBSERVATION_REPRESENTATION_F1_VECTOR:
        raise RuntimeError(f"{path} is not an F1TENTH (f1_vector) world model")
    if cfg.get("reward_factor_kind") != REWARD_FACTOR_KIND_F1TENTH:
        raise RuntimeError(f"{path}: expected the F1TENTH reward factors")
    model = ChunkWorldModel(
        d_model=int(cfg["d_model"]), d_spatial=int(cfg["d_spatial"]), n_spatial=int(cfg["n_spatial"]),
        horizon=int(cfg["horizon"]), action_dim=int(cfg["action_dim"]), model_action_dim=int(cfg["model_action_dim"]),
        n_register=int(cfg["n_register"]), n_agent=int(cfg["n_agent"]), n_heads=int(cfg["n_heads"]),
        depth=int(cfg["depth"]), policy_depth=int(cfg["policy_depth"]), dropout=0.0,
        mlp_ratio=float(cfg["mlp_ratio"]), time_every=int(cfg["time_every"]), space_mode=str(cfg["space_mode"]),
        n_tasks=int(cfg["n_tasks"]), reward_num_bins=int(cfg["num_reward_bins"]),
        reward_log_low=float(cfg["reward_log_low"]), reward_log_high=float(cfg["reward_log_high"]),
        value_num_bins=int(cfg["num_value_bins"]), value_log_low=float(cfg["value_log_low"]),
        value_log_high=float(cfg["value_log_high"]), policy_log_std_init=float(cfg["policy_log_std_init"]),
        use_states=bool(cfg["use_states"]), state_dim=int(cfg["state_dim"]), use_agent_heads=True,
        scorer_arch=str(cfg["scorer_arch"]), dynamics_arch=str(cfg["dynamics_arch"]),
        latent_codec_arch=str(cfg["latent_codec_arch"]), observation_representation=OBSERVATION_REPRESENTATION_F1_VECTOR,
        value_num_heads=int(cfg["value_num_heads"]), scene_factor_dim=int(cfg.get("scene_factor_dim", 0)),
        reward_factor_kind=REWARD_FACTOR_KIND_F1TENTH, md_obs_dim=int(cfg.get("md_obs_dim", 144)),
    ).to(device)
    model.load_state_dict(ckpt["chunk_world_model"], strict=True)
    prior = nn.ModuleDict({
        "policy_transformer": copy.deepcopy(model.policy_transformer),
        "policy_head": copy.deepcopy(model.policy_head),
        "policy_logstd_head": copy.deepcopy(model.policy_logstd_head),
    }).to(device)
    scorers = nn.ModuleDict({
        "reward_head": copy.deepcopy(model.reward_head),
        "reward_scorer": copy.deepcopy(model.reward_scorer),
        "reward_factor_head": copy.deepcopy(model.reward_factor_head),
        "continuation_head": copy.deepcopy(model.continuation_head),
    }).to(device)
    value_head = copy.deepcopy(model.value_heads).to(device)
    head_cfg = dict(ckpt["risk_head_config"])
    if tuple(head_cfg.get("events", ())) != F1_EVENTS:
        raise RuntimeError(f"{path}: risk-head events {head_cfg.get('events')} are not {F1_EVENTS}")
    risk_head = RiskHead.from_config(head_cfg).to(device)
    risk_head.load_state_dict(ckpt["risk_head"], strict=True)
    for m in (model, prior, scorers, value_head, risk_head):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return dict(model=model, prior=prior, scorers=scorers, value_head=value_head, risk_head=risk_head, config=cfg,
                log_std_min=float(cfg["policy_log_std_min"]), log_std_max=float(cfg["policy_log_std_max"]))


# ------------------------------------------------------------------ planner
class SaferPlanner:
    """act(obs) -> (alpha, delta) for one car; reset() at the start of every run."""

    def __init__(self, checkpoint: str, device: str = "cuda:0", risk_weight: float = 50.0,
                 config: Optional[dict] = None):
        self.device = torch.device(device)
        self.m = load_checkpoint(checkpoint, self.device)
        self.cfg = dict(MPPI_CONFIG, **(config or {}))
        if abs(float(self.m["config"]["gamma"]) - float(self.cfg["gamma"])) > 1e-12:
            raise RuntimeError("the checkpoint's gamma differs from the planner's")
        self.risk_weight = float(risk_weight)
        use_mp = self.device.type == "cuda"
        self.amp = dict(device_type=self.device.type, enabled=use_mp,
                        dtype=torch.bfloat16 if use_mp and torch.cuda.is_bf16_supported() else torch.float16)
        self.reset()

    def reset(self) -> None:
        self.z_hist: list = []
        self.a_hist: list = []
        self.prev_mean: Optional[torch.Tensor] = None
        self.last: dict = {}

    @torch.inference_mode()
    def act(self, obs: np.ndarray, controller: str = "safer") -> np.ndarray:
        """obs: the 144-value observation (observation.py). controller: 'safer' (MPPI) or 'bc_prior'."""
        model, c = self.m["model"], self.cfg
        ctx, H, A = int(c["context_length"]), int(c["horizon"]), int(c["action_dim"])
        z = model.encode_md_observations(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32,
                                                         device=self.device)[:, None])
        self.z_hist.append(z[0:1])
        del self.z_hist[:-ctx]
        z_ctx, ctx_act = left_padded_context(self.z_hist, self.a_hist, ctx, int(model.model_action_dim))
        z_ctx, ctx_act = z_ctx.float(), ctx_act.float()
        task = torch.zeros(1, dtype=torch.long, device=self.device)
        if controller == "safer":
            action = self._mppi(z_ctx, ctx_act, task)
        elif controller == "bc_prior":
            out = model.predict_policy(z_ctx.float(), ctx_act.float(), task, state_context=None,
                                       policy_modules=self.m["prior"])
            action = torch.tanh(out["policy_mean_pre"].float()[:, 0, :A])
        else:
            raise ValueError(f"controller must be 'safer' or 'bc_prior', got {controller!r}")
        action = action.float().clamp(-1.0, 1.0)
        self.a_hist.append(action[0].detach().clone())
        del self.a_hist[:-ctx]
        return action[0].cpu().numpy().astype(np.float32)

    # train_online_loop._mppi_plan_actions (eager path), one car
    def _mppi(self, z_ctx, ctx_act, task):
        cm, c = self.m["model"], self.cfg
        B, H, A = 1, int(c["horizon"]), int(c["action_dim"])
        dev = self.device
        with torch.amp.autocast(**self.amp):
            planning_context = cm.compress_latents(z_ctx)
            context_memory = cm.encode_context(z_ctx.float(), ctx_act.float(), task, state_context=None,
                                               planning_context=planning_context)
            out = cm.predict_policy(z_ctx.float(), ctx_act.float(), task, state_context=None,
                                    policy_modules=self.m["prior"], context_memory=context_memory)
            bc_mean_pre = out["policy_mean_pre"].float()[:, :H, :A]
        bc_policy_actions = torch.tanh(bc_mean_pre[:, :H]).clamp(-0.999, 0.999).contiguous()
        mean = bc_policy_actions.clone()
        if bool(c["warm_start"]) and isinstance(self.prev_mean, torch.Tensor) and tuple(self.prev_mean.shape) == (H, A):
            shifted = torch.roll(self.prev_mean.to(device=dev, dtype=mean.dtype), shifts=-1, dims=0)
            shifted[-1] = bc_policy_actions[0, -1]
            mean[0] = _bound_mppi_action_sequences(shifted.unsqueeze(0))[0]
        base_std = torch.zeros(A, device=dev, dtype=mean.dtype)
        base_std[0] = float(c["init_acc_std"])
        base_std[1] = float(c["init_steer_std"])
        std = base_std.view(1, 1, A).expand(B, H, A).clone()
        min_std = torch.zeros_like(base_std)
        min_std[0] = float(c["min_acc_std"])
        min_std[1] = float(c["min_steer_std"])
        min_std = min_std.view(1, 1, A)
        num_samples = max(1, int(c["k"]))
        mix = max(0.0, float(c["mixture_coef"]))
        num_base = int(round(num_samples * mix))
        num_base = max(1, num_base) if mix > 0 else 0
        num_elites = max(1, int(c["num_elites"]))
        temperature = float(c["temperature"])
        momentum = min(1.0, max(0.0, float(c["momentum"])))
        fixed_bc_candidates = None
        if num_base > 0:
            bc_std = torch.zeros(A, device=dev, dtype=mean.dtype)
            bc_std[0] = float(c["base_acc_std"])
            bc_std[1] = float(c["base_steer_std"])
            bc_noise = torch.randn(B, num_base, H, A, device=dev, dtype=mean.dtype) * bc_std.view(1, 1, 1, A)
            fixed_bc_candidates = _bound_mppi_action_sequences(bc_policy_actions[:, None] + bc_noise)
        for _iter in range(max(1, int(c["iterations"]))):
            noise = torch.randn(B, num_samples, H, A, device=dev, dtype=mean.dtype)
            parts = [_bound_mppi_action_sequences(mean[:, None] + noise * std[:, None])]
            if fixed_bc_candidates is not None:
                parts.append(fixed_bc_candidates)
            parts.append(bc_policy_actions.clamp(-0.999, 0.999)[:, None])
            candidates = torch.cat(parts, dim=1).contiguous()
            scores, returns, rewards, risk = self._score(z_ctx, ctx_act, task, candidates, context_memory,
                                                         planning_context)
            K_total = int(candidates.shape[1])
            E = min(num_elites, K_total)
            elite_scores, elite_idx = torch.topk(scores, E, dim=1)
            elite_actions = candidates.gather(1, elite_idx[:, :, None, None].expand(-1, -1, H, A))
            centered = elite_scores - elite_scores.max(dim=1, keepdim=True).values
            weights = torch.softmax(temperature * centered, dim=1)
            new_mean = (weights[:, :, None, None] * elite_actions).sum(dim=1)
            new_mean = _bound_mppi_action_sequences(new_mean)
            new_std = (weights[:, :, None, None] * (elite_actions - new_mean[:, None]).pow(2)).sum(dim=1).sqrt()
            new_std = new_std.clamp_min(min_std)
            mean = momentum * mean + (1.0 - momentum) * new_mean
            mean = _bound_mppi_action_sequences(mean)
            std = new_std
        self.prev_mean = mean[0].detach().cpu()
        self.last = dict(plan=mean[0].detach().cpu().numpy(), prior=bc_policy_actions[0].detach().cpu().numpy(),
                         best_score=float(elite_scores[0, 0]), prior_score=float(scores[0, -1]),
                         risk_best=float(risk[0, int(elite_idx[0, 0])]))
        return mean[:, 0]

    # train_online_loop._mppi_score_candidates
    @torch.no_grad()
    def _score(self, z_ctx, ctx_act, task_ids_ctx, candidates, context_memory, planning_context):
        cm, c = self.m["model"], self.cfg
        dev = self.device
        B, K, H, A = candidates.shape
        z_source = z_ctx if planning_context is None else planning_context
        z_rep = z_source[:, None].expand(B, K, *z_source.shape[1:]).reshape(B * K, *z_source.shape[1:]).contiguous()
        act_rep = ctx_act[:, None].expand(B, K, *ctx_act.shape[1:]).reshape(B * K, *ctx_act.shape[1:]).contiguous()
        task_rep = task_ids_ctx[:, None].expand(B, K).reshape(B * K).contiguous()
        flat_actions = candidates.reshape(B * K, H, A).contiguous()
        expanded_memory = (context_memory[:, None].expand(B, K, *context_memory.shape[1:])
                           .reshape(B * K, *context_memory.shape[1:]).contiguous())
        model_action_dim = int(getattr(cm, "model_action_dim", 16))
        fut_actions = torch.zeros(B * K, H, model_action_dim, device=dev, dtype=flat_actions.dtype)
        fut_actions[:, :, :A] = flat_actions
        with torch.amp.autocast(**self.amp):
            out = cm(z_rep, act_rep, fut_actions, task_rep, compute_policy=False, scorer_modules=self.m["scorers"],
                     state_context=None, context_memory=expanded_memory,
                     planning_context=(z_rep if planning_context is not None else None), decode_latents=False)
        rewards = _blended_predicted_rewards(out, flat_actions, act_rep[:, -1:, 1])
        continuations = torch.sigmoid(out["cont_logits"].float()).clamp(0.0, 1.0)
        gamma = float(c["gamma"])
        discounts = torch.pow(torch.full((H,), gamma, device=dev, dtype=rewards.dtype),
                              torch.arange(H, device=dev, dtype=rewards.dtype))
        survival_before = torch.cat([torch.ones(continuations.shape[0], 1, device=continuations.device,
                                                dtype=continuations.dtype),
                                     torch.cumprod(continuations[:, :-1], dim=1)], dim=1)
        reward_returns = (rewards * discounts * survival_before).sum(dim=1)
        value_weight = float(c["value_score_weight"])
        value_logits_all, value_centers = value_ensemble_logits(self.m["value_head"], out["value_features"][:, -1:])
        value_mean, value_std = value_ensemble_stats(value_logits_all, value_centers)
        terminal_values = value_mean[:, 0]
        terminal_survival = torch.prod(continuations, dim=1)
        returns = reward_returns + value_weight * (gamma ** H) * terminal_survival * terminal_values
        returns = returns.reshape(B, K)
        rewards = rewards.reshape(B, K, H)
        risk = torch.zeros_like(returns)
        if self.risk_weight > 0.0:
            risk = self.m["risk_head"].penalty(z_rep, out["h"]).reshape(B, K)
            returns = returns - self.risk_weight * risk.to(dtype=returns.dtype)
        scores = returns
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite MPPI scores")
        return scores, returns, rewards, risk
