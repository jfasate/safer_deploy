# model.py
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    """
    videos: (B,T,C,H,W) float in [0,1]
    returns: (B,T,Np,Dp) where Dp = patch*patch*C and Np = (H/patch)*(W/patch)
    """
    assert videos_btchw.dim() == 5
    B, T, C, H, W = videos_btchw.shape
    assert H % patch == 0 and W % patch == 0
    x = videos_btchw.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch)          # (BT, C*pp, Np)
    cols = cols.transpose(1, 2).contiguous()                     # (BT, Np, Dp)
    Np, Dp = cols.shape[1], cols.shape[2]
    return cols.reshape(B, T, Np, Dp)


def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """
    patches: (B,T,Np,Dp) -> (B,T,C,H,W)
    """
    assert patches_btnd.dim() == 4
    B, T, Np, Dp = patches_btnd.shape
    assert Dp == C * patch * patch
    x = patches_btnd.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()  # (BT, Dp, Np)
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)  # (BT, C, H, W)
    return out.reshape(B, T, C, H, W)


def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    # fp32 by construction
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n,1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)  # (1,d)
    k   = torch.floor(i / 2.0)
    # stable: exp(log(base) * exponent)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))  # (n,d) fp32


def add_sinusoidal_positions(tokens_btSd: torch.Tensor) -> torch.Tensor:
    B, T, S, D = tokens_btSd.shape
    device = tokens_btSd.device
    pos_t = sinusoid_table(T, D, device=device)  # fp32
    pos_s = sinusoid_table(S, D, device=device)  # fp32
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :]) * (1.0 / math.sqrt(D))
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


class MAEReplacer(nn.Module):
    def __init__(
        self,
        d_model: int,
        p_min: float = 0.0,
        p_max: float = 0.9,
        fg_beta: float = 0.0,
        fg_tau: float = 0.02,
    ):
        """
        fg_beta: fraction of masked patches forced to be foreground (0 = disabled).
        fg_tau:  threshold on fg_ratio to consider a patch foreground.
        """
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.fg_beta = float(fg_beta)
        self.fg_tau = float(fg_tau)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        patches_btnd: torch.Tensor,                  # (B,T,Np,D)
        fg_ratio_btNp1: Optional[torch.Tensor] = None,  # (B,T,Np,1) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B,T,Np,D)
        returns:
          replaced: (B,T,Np,D)
          mae_mask: (B,T,Np,1) bool, True where masked (must reconstruct)
          keep_prob:(B,T,1) float
        """
        B, T, Np, D = patches_btnd.shape
        device = patches_btnd.device

        # fast path: deterministic "no MAE"
        if self.p_min == 0.0 and self.p_max == 0.0:
            keep_prob = torch.ones((B, T, 1), device=device, dtype=patches_btnd.dtype)
            mae_mask = torch.zeros((B, T, Np, 1), device=device, dtype=torch.bool)
            return patches_btnd, mae_mask, keep_prob

        p_bt = torch.empty((B, T), device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p_bt).unsqueeze(-1)  # (B,T,1)

        use_fg = (
            fg_ratio_btNp1 is not None
            and self.fg_beta > 0.0
            and fg_ratio_btNp1.shape[:3] == (B, T, Np)
        )

        if not use_fg:
            # Original random masking
            keep = (torch.rand((B, T, Np), device=device) < keep_prob)  # (B,T,Np)
        else:
            fg_ratio = fg_ratio_btNp1[..., 0].to(device=device)  # (B,T,Np)
            fg_mask = fg_ratio > self.fg_tau                      # (B,T,Np) bool
            keep = torch.ones((B, T, Np), device=device, dtype=torch.bool)
            idx_all = torch.arange(Np, device=device)
            for b in range(B):
                for t in range(T):
                    K = int(round(p_bt[b, t].item() * Np))  # number of masked patches
                    if K <= 0:
                        continue
                    K_fg = int(round(self.fg_beta * K))

                    masked_idx = []
                    fg_idx = idx_all[fg_mask[b, t]]
                    if fg_idx.numel() > 0 and K_fg > 0:
                        perm_fg = torch.randperm(fg_idx.numel(), device=device)
                        k = min(K_fg, fg_idx.numel())
                        masked_idx.append(fg_idx[perm_fg[:k]])

                    used_fg = masked_idx[-1].numel() if masked_idx else 0
                    K_rest = K - used_fg
                    if K_rest > 0:
                        perm_all = torch.randperm(Np, device=device)
                        masked_idx.append(idx_all[perm_all[:K_rest]])

                    if masked_idx:
                        m = torch.cat(masked_idx, dim=0).unique()
                        keep[b, t, m] = False

        keep_ = keep.unsqueeze(-1)
        mask_tok = self.mask_token.to(dtype=patches_btnd.dtype)
        replaced = torch.where(keep_, patches_btnd, mask_tok.view(1, 1, 1, D))
        mae_mask = (~keep_).to(torch.bool)
        return replaced, mae_mask, keep_prob


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class AdaLNModulation(nn.Module):
    """Produce (scale, shift, gate) triplets per sub-layer from a conditioning vector.

    Initialised near a regular transformer block:
      norm:     norm(x) * (1+small_action_scale) + small_action_shift
      residual: (1+small_action_gate) * sublayer(h)

    Fully zero initialisation makes the AdaLN branch exactly action-independent
    at step 0. That is good for adding conditioning to a pretrained denoiser,
    but here the dynamics model is trained from scratch and we need the action
    conditioning path to be live from the first update.
    """
    def __init__(self, d_model: int, n_outputs: int):
        super().__init__()
        self.n_outputs = n_outputs
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_outputs * d_model),
        )
        nn.init.normal_(self.proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.proj[-1].bias)
        # Outputs are triplets: scale, shift, gate for each sublayer.
        with torch.no_grad():
            bias = self.proj[-1].bias.view(self.n_outputs, d_model)
            bias[2::3].fill_(1.0)

    def forward(self, c: torch.Tensor) -> tuple:
        """c: (B, T, D) -> tuple of n_outputs tensors, each (B, T, D)."""
        return self.proj(c).chunk(self.n_outputs, dim=-1)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x_nld: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        """
        x: (N,L,D)
        attn_mask: bool, True means "allowed to attend" (for torch SDPA), broadcastable to (N,1,L,L) or (N,H,L,L)
        """
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)

        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)                               # (S,S) True=allowed
        attn_mask = allow.unsqueeze(0).unsqueeze(0)                # (1,1,S,S) True=allowed (PyTorch SDPA bool mask)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # full mixing across modalities
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            # non-agent tokens: can attend to everything EXCEPT agent tokens
            # agent tokens: attend only to agent tokens (keeps them inert in pretrain)
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow = torch.ones((S, S), dtype=torch.bool, device=device)

            # non-agent queries cannot see agent keys
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)

            # agent queries only see agent keys
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        if self.mode == "wm_agent_cautious":
            # Action-conditioned WM: non-agent tokens read actions, shortcuts, latents, etc., but never AGENT.
            # Agent readout may attend to SPATIAL / PROPRIO / REGISTER / AGENT only — not raw ACTION or shortcut tokens.
            is_q_agent = q_mod == int(Modality.AGENT)
            is_k_agent = k_mod == int(Modality.AGENT)
            is_k_spatial = k_mod == int(Modality.SPATIAL)
            is_k_proprio = k_mod == int(Modality.PROPRIO)
            is_k_register = k_mod == int(Modality.REGISTER)

            allow_non_agent_q = ~is_k_agent
            # SPATIAL | PROPRIO | REGISTER | AGENT only (excludes ACTION, SHORTCUT_*).
            allow_agent_q = is_k_spatial | is_k_proprio | is_k_register | is_k_agent

            return torch.where(is_q_agent, allow_agent_q, allow_non_agent_q)

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        if self.latents_only:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]  # (B,T,L,D)
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat_nld, is_causal=True)
            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x_btSd.clone()
            x[:, :, :L, :] = out
            return x
        else:
            x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
            out = self.attn(x_nld, is_causal=True)
            return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)
        self.use_adaln = bool(use_adaln)

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents)
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

        if self.use_adaln:
            # (scale, shift, gate) per sublayer: 6 without time, 9 with time
            n_params = 9 if self.do_time else 6
            self.adaln = AdaLNModulation(d_model, n_params)
            # Agent tokens are action-blind: no AdaLN modulation, standard residual.
            non_agent = (modality_ids != int(Modality.AGENT))  # (S,)
            self.register_buffer("non_agent_mask", non_agent[None, None, :, None], persistent=False)  # (1,1,S,1)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_adaln and cond is not None:
            # m: (1,1,S,1) bool — True for non-agent positions.
            # Non-agent tokens get action-conditioned AdaLN-Zero (scale, shift, gate).
            # Agent tokens see plain norm and standard residual (action-blind readout).
            m = self.non_agent_mask
            if self.do_time:
                s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = self.adaln(cond)
            else:
                s1, sh1, g1, s3, sh3, g3 = self.adaln(cond)

            # -- space sublayer --
            h = self.norm1(x) * (1 + s1[:, :, None, :] * m) + sh1[:, :, None, :] * m
            r = self.drop1(self.space(h))
            x = x + r * (g1[:, :, None, :] * m + ~m)  # non-agent: gate*r, agent: 1*r

            # -- time sublayer --
            if self.do_time:
                h = self.norm2(x) * (1 + s2[:, :, None, :] * m) + sh2[:, :, None, :] * m
                r = self.drop2(self.time(h))
                x = x + r * (g2[:, :, None, :] * m + ~m)

            # -- mlp sublayer --
            h = self.norm3(x) * (1 + s3[:, :, None, :] * m) + sh3[:, :, None, :] * m
            r = self.drop3(self.mlp(h))
            x = x + r * (g3[:, :, None, :] * m + ~m)
        else:
            x = x + self.drop1(self.space(self.norm1(x)))
            if self.do_time:
                x = x + self.drop2(self.time(self.norm2(x)))
            x = x + self.drop3(self.mlp(self.norm3(x)))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, n_latents=n_latents,
                modality_ids=modality_ids, space_mode=space_mode,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
                latents_only_time=latents_only_time,
                use_adaln=use_adaln,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond=cond)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
        mae_fg_beta: float = 0.0,
        mae_fg_tau: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids()  # CPU buffer, moves with .to(device)

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="encoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
        )
        self.mae = MAEReplacer(
            d_model=d_model,
            p_min=mae_p_min,
            p_max=mae_p_max,
            fg_beta=mae_fg_beta,
            fg_tau=mae_fg_tau,
        )

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(
        self,
        patch_tokens_btnd: torch.Tensor,
        fg_ratio_btNp1: Optional[torch.Tensor] = None,
    ):
        B, T, Np, Dp = patch_tokens_btnd.shape
        assert Np == self.n_patches

        proj = self.patch_proj(patch_tokens_btnd)                           # (B,T,Np,D)
        proj_masked, mae_mask, keep_prob = self.mae(proj, fg_ratio_btNp1)   # (B,T,Np,D), (B,T,Np,1), (B,T,1)

        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)        # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)

        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :]))
        return z, (mae_mask, keep_prob)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        d_patch: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        output_logits: bool = False,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.output_logits = output_logits

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(d_model, d_patch)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="decoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
        )

    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        assert L == self.n_latents

        lat = torch.tanh(self.up_proj(z_btLd))                                 # (B,T,L,D)
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)                                  # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)

        x = self.transformer(tokens)
        x_p = x[:, :, L:, :]
        out = self.patch_head(x_p)
        return out if self.output_logits else torch.sigmoid(out)                # (B,T,Np,Dp)


class BEVPointQueryHead(nn.Module):
    """
    Predicts a fixed set of BEV actor points (class + normalized xy) from encoder latents.
    Used with DETR-style Hungarian matching (no ordering). xy in [0,1]^2 via sigmoid.
    """

    def __init__(
        self,
        d_bottleneck: int,
        num_queries: int,
        num_fg_classes: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_fg_classes = int(num_fg_classes)
        self.num_classes = int(num_fg_classes) + 1  # + no-object
        h = int(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(int(d_bottleneck), h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.fc_xy = nn.Linear(h, self.num_queries * 2)
        self.fc_cls = nn.Linear(h, self.num_queries * self.num_classes)

    def forward(self, z_btld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, L, _ = z_btld.shape
        g = self.mlp(z_btld.mean(dim=2))
        xy = self.fc_xy(g).view(B, T, self.num_queries, 2).sigmoid()
        logits = self.fc_cls(g).view(B, T, self.num_queries, self.num_classes)
        return logits, xy


class Tokenizer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, point_head: Optional[BEVPointQueryHead] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.point_head = point_head

    def forward(
        self,
        patches_btnd: torch.Tensor,
        fg_ratio_btNp1: Optional[torch.Tensor] = None,
    ):
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd, fg_ratio_btNp1)
        pred = self.decoder(z)
        if self.point_head is None:
            return pred, mae_mask, keep_prob
        plogits, pxy = self.point_head(z)
        return pred, mae_mask, keep_prob, plogits, pxy


def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """
    z: (B,T,L,D_b) where L == n_spatial * k
    -> (B,T,n_spatial,D_b*k)
    """
    B, T, L, D = z_btLd.shape
    assert L == n_spatial * k, f"L={L} must equal n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z_btSd: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    z: (B,T,n_spatial,D_b*k) -> (B,T,n_spatial*k,D_b)
    """
    B, T, S, DK = z_btSd.shape
    assert DK % k == 0, f"D={DK} must be divisible by k={k}"
    D = DK // k
    return z_btSd.view(B, T, S * k, D)


class ActionEncoder(nn.Module):
    """
    Continuous actions in [-1,1], shape (B,T,A) -> token (B,T,1,D).
    If actions is None (unlabeled pretrain), emits a learned base token.
    """
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)

        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.normal_(self.fc2.weight, std=1e-2)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],                 # (B,T,A) or None
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,         # (B,T,A) or (A,)
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions
            if act_mask is not None:
                x = x * act_mask
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out[:, :, None, :] if as_tokens else out


class StateEncoder(nn.Module):
    """
    Privileged state vector (B,T,S) -> token (B,T,1,D).
    If state_vec is None (e.g. inference without privileged info), emits a learned base token.
    """
    def __init__(self, d_model: int, state_dim: int = 7, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)
        self.fc1 = nn.Linear(self.state_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        state_vec: Optional[torch.Tensor],          # (B,T,S) or None
        *,
        batch_time_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if state_vec is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            out = self.fc2(F.silu(self.fc1(state_vec))) + self.base.view(1, 1, -1)
        return out[:, :, None, :]  # (B,T,1,d_model)


# ---------------------------------------------------------------------------
# Privileged state normalization constants.
#
# Computed once from the full training corpus (2,458,128 rows across 520 demo
# files) by `dreamer4/compute_state_vec_stats.py`; see
# `dreamer4/state_vec_scales.json`. Layout mirrors
# `eval_dreamer4._extract_state_vec`: [acc_x, acc_y, lateral, vel_x, vel_y,
# yaw_rate_deg_s, tl_ttc].
#
# These balance both state-token inputs and StatePredictorHead targets in the
# chunk world model (otherwise yaw_rate, whose std is about 375 deg/s, would
# dominate). Legacy denoising paths may still normalize at their own boundary.
# ---------------------------------------------------------------------------
STATE_MEAN_7D: Tuple[float, ...] = (
    -0.005327,  # acc_x  (m/s^2)
     0.01853,   # acc_y  (m/s^2)
    -0.004803,  # lateral_offset (m, + = right)
     3.035,     # vel_x  (m/s)
     0.006624,  # vel_y  (m/s)
    12.82,      # yaw_rate (deg/s)
     0.1104,    # tl_ttc (s, -1 if none)
)
STATE_STD_7D: Tuple[float, ...] = (
    1.558,
    0.9352,
    0.2415,
    2.617,
    0.1929,
    374.8,
    2.055,
)


def _as_1d_tensor(v: Tuple[float, ...], ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)


def normalize_state_vec(
    x: torch.Tensor,
    *,
    mean: Tuple[float, ...] = STATE_MEAN_7D,
    std: Tuple[float, ...] = STATE_STD_7D,
) -> torch.Tensor:
    """z-score normalize a raw `state_vec` (…,S) to the predictor's target space."""
    D = x.shape[-1]
    m = _as_1d_tensor(mean[:D], x)
    s = _as_1d_tensor(std[:D], x).clamp_min(1e-6)
    return (x - m) / s


def denormalize_state_vec(
    z: torch.Tensor,
    *,
    mean: Tuple[float, ...] = STATE_MEAN_7D,
    std: Tuple[float, ...] = STATE_STD_7D,
) -> torch.Tensor:
    """Inverse of `normalize_state_vec`: z-score (…,S) back to raw units."""
    D = z.shape[-1]
    m = _as_1d_tensor(mean[:D], z)
    s = _as_1d_tensor(std[:D], z)
    return z * s + m


# Reward factors are nonnegative physical magnitudes. Normalize speed and the
# terminal-penalty magnitude to the 6 m/s task scale, while keeping lateral
# distance and heading error in their native meter/radian units.
REWARD_FACTOR_SCALE: Tuple[float, ...] = (6.0, 6.0, 1.0, 1.0, 6.0)


def normalize_reward_factors(x: torch.Tensor) -> torch.Tensor:
    """Scale four physical factors and an optional terminal-penalty magnitude."""
    scale = _as_1d_tensor(REWARD_FACTOR_SCALE[: x.shape[-1]], x)
    return x / scale


def denormalize_reward_factors(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized reward factors back to physical units."""
    scale = _as_1d_tensor(REWARD_FACTOR_SCALE[: x.shape[-1]], x)
    return x * scale


class StatePredictorHead(nn.Module):
    """Predict state_vec[t+1] from h_pooled[t], in z-score space.

    During imagination the dynamics lacks ground-truth state_vec (velocity,
    lateral offset, …).  This head provides a learned estimate so that the
    StateEncoder inside Dynamics receives informative input instead of a
    constant base token, keeping reward-head predictions meaningful.

    The target/output space is z-score normalized using `STATE_MEAN_7D` and
    `STATE_STD_7D` so the MSE loss is balanced across dimensions of very
    different magnitudes (yaw_rate std ≈ 375 deg/s vs. lateral std ≈ 0.24 m).
    Callers that feed predictions back into Dynamics must apply
    `denormalize_state_vec(.)` first — the Dynamics's StateEncoder still
    expects raw units.
    """

    def __init__(self, d_model: int, state_dim: int = 7, hidden_mult: float = 2.0):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        # small init for last layer so predictions start near zero (i.e. at
        # the dataset mean in raw space after denormalization)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> (..., state_dim) in z-score space."""
        return self.net(h)


class InverseDynamicsHead(nn.Module):
    """Auxiliary head: predicts action[t] from (z_hat[t-1], z_hat[t]).

    Used as an inverse-dynamics auxiliary loss during dynamics training.
    Forces the model to encode action information in consecutive latent
    transitions, directly combating action-insensitivity / ego freeze.

    Args:
        n_spatial:   number of packed spatial slots (Sz)
        d_spatial:   dimension of each packed spatial slot (Dz = d_bottleneck * packing_factor)
        action_dim:  action vector size (default 16)
        hidden:      MLP hidden size
    """
    def __init__(self, n_spatial: int, d_spatial: int, action_dim: int = 16, hidden: int = 256):
        super().__init__()
        in_dim = 2 * n_spatial * d_spatial
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Tanh(),
        )
        nn.init.normal_(self.net[1].weight, std=0.02)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[3].weight, std=0.02)
        nn.init.zeros_(self.net[3].bias)
        nn.init.normal_(self.net[5].weight, std=0.02)
        nn.init.zeros_(self.net[5].bias)

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        """
        z_prev, z_curr: (..., Sz, Dz) — any leading batch/time dims.
        Returns: (..., action_dim) predicted action.
        """
        shape = z_prev.shape[:-2]
        inp = torch.cat([z_prev.reshape(*shape, -1), z_curr.reshape(*shape, -1)], dim=-1)
        return self.net(inp)


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_agent: int = 1,
        use_ids: bool = True,
        n_tasks: int = 128,
        d_task: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_agent = int(n_agent)
        self.use_ids = bool(use_ids)
        self.n_tasks = int(n_tasks)
        self.d_task = int(d_task)

        if self.use_ids:
            self.task_table = nn.Embedding(self.n_tasks, self.d_model)
        else:
            self.task_proj = nn.Linear(self.d_task, self.d_model)

        self.agent_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.agent_base, std=0.02)

    def forward(self, task: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if self.use_ids:
            emb = self.task_table(task.to(torch.long))  # (B,D)
        else:
            emb = self.task_proj(task)                  # (B,D)

        x = emb + self.agent_base.view(1, -1)          # (B,D)
        return x[:, None, None, :].expand(B, T, self.n_agent, self.d_model)


class PolicyHeadMTP(nn.Module):
    """Multi-token action prediction from agent readouts h_t.
    Input:  h_t (B, T, D) -- agent readouts (pool n_agent first if needed)
    Output: (B, T, L, action_dim) for continuous; (B, T, L, num_classes) logits for categorical.
    """
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        L: int = 2,
        kind: str = "continuous",  # "continuous" or "categorical"
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        self.L = int(L)
        self.kind = str(kind)
        self.projector = MLP(d_model=self.d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        out_dim = self.action_dim
        self.out = nn.Linear(self.d_model, self.L * out_dim)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        # h_t: (B, T, D)
        x = self.projector(h_t)
        out = self.out(x)  # (B, T, L*out_dim)
        return out.view(*out.shape[:2], self.L, self.action_dim)


class VectorHeadMTP(nn.Module):
    """Multi-token vector prediction from temporal readouts.

    This is the non-distributional counterpart of ``RewardHeadMTP`` for
    vector-valued regression targets or independent binary-event logits.
    Output activations are deliberately left to the caller.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        L: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(output_dim)
        self.L = int(L)
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.out = nn.Linear(self.d_model, self.L * self.output_dim)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        x = self.projector(h_t)
        out = self.out(x)
        return out.view(*out.shape[:2], self.L, self.output_dim)


class RewardHeadMTP(nn.Module):
    """Multi-token reward prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, L, K), centers_log (K,)
    """
    def __init__(
        self,
        d_model: int,
        L: int = 2,
        num_bins: int = 101,
        log_low: float = -3.0,
        log_high: float = 3.0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L = int(L)
        self.num_bins = int(num_bins)
        self.projector = MLP(d_model=self.d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(self.d_model, self.L * self.num_bins)
        centers = torch.linspace(log_low, log_high, self.num_bins)
        self.register_buffer("centers_log", centers)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.projector(h_t)
        out = self.out(x)  # (B, T, L*K)
        logits = out.view(*out.shape[:2], self.L, self.num_bins)
        return logits, self.centers_log


class ContinuationHeadMTP(nn.Module):
    """Multi-token continuation prediction.

    Input:  h_t (B, T, D)
    Output: logits (B, T, L), where sigmoid(logits) predicts c_t in [0, 1].
    """
    def __init__(
        self,
        d_model: int,
        L: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L = int(L)
        self.projector = MLP(d_model=self.d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(self.d_model, self.L)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        x = self.projector(h_t)
        out = self.out(x)
        return out.view(*out.shape[:2], self.L)


class ValueHeadMTP(nn.Module):
    """Value head with symlog twohot bins (same structure as RewardHeadMTP).
    Input:  h_t (B, T, D)
    Output: logits (B, T, L, K), centers_log (K,)
    """
    def __init__(
        self,
        d_model: int,
        L: int = 2,
        num_bins: int = 101,
        log_low: float = -10.0,
        log_high: float = 10.0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L = int(L)
        self.num_bins = int(num_bins)
        self.projector = MLP(d_model=self.d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(self.d_model, self.L * self.num_bins)
        centers = torch.linspace(log_low, log_high, self.num_bins)
        self.register_buffer("centers_log", centers)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.projector(h_t)
        out = self.out(x)
        logits = out.view(*out.shape[:2], self.L, self.num_bins)
        return logits, self.centers_log


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",  # wm_agent | wm_agent_isolated | wm_agent_cautious
        use_states: bool = False,
        state_dim: int = 7,
        use_adaln: bool = False,
    ):
        super().__init__()
        assert d_spatial % d_bottleneck == 0, "expected packing: d_spatial = d_bottleneck * packing_factor"
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.n_register = int(n_register)
        self.n_agent = int(n_agent)
        self.k_max = int(k_max)
        self.use_adaln = bool(use_adaln)

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        self.use_states = bool(use_states)
        if self.use_states:
            self.state_encoder = StateEncoder(d_model=self.d_model, state_dim=int(state_dim))

        # shortcut conditioning
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model)

        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.use_states:
            segments.insert(1, (Modality.PROPRIO, 1))
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=int(n_heads),
            depth=int(depth),
            n_latents=0,
            modality_ids=modality_ids,
            space_mode=space_mode,
            dropout=float(dropout),
            mlp_ratio=float(mlp_ratio),
            time_every=int(time_every),
            latents_only_time=False,
            use_adaln=use_adaln,
        )

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B,T,16) or None
        step_idxs: torch.Tensor,                  # (B,T)
        signal_idxs: torch.Tensor,                # (B,T)
        packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
        agent_tokens: Optional[torch.Tensor] = None,
        state_vec: Optional[torch.Tensor] = None,  # (B,T,S) or None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B,T,n_spatial,d_model)

        action_tokens = self.action_encoder(
            actions,
            batch_time_shape=(B, T),
            act_mask=act_mask,
            as_tokens=True,
        )  # (B,T,1,d_model)

        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)

        step_tok = self.step_embed(step_idxs.to(torch.long))[:, :, None, :]
        sig_tok = self.signal_embed(signal_idxs.to(torch.long))[:, :, None, :]

        if self.use_states:
            state_tokens = self.state_encoder(state_vec, batch_time_shape=(B, T))  # (B,T,1,d_model)
            base_toks = [action_tokens, state_tokens, sig_tok, step_tok, spatial_tokens, reg]
        else:
            base_toks = [action_tokens, sig_tok, step_tok, spatial_tokens, reg]

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros((B, T, self.n_agent, self.d_model), device=spatial_tokens.device, dtype=spatial_tokens.dtype)
            toks = base_toks + [agent_tokens]
        else:
            toks = base_toks

        tokens = torch.cat(toks, dim=2)  # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)
        # AdaLN: use action embedding as conditioning for every layer norm
        act_cond = action_tokens.squeeze(2) if self.use_adaln else None  # (B,T,d_model)
        x = self.transformer(tokens, cond=act_cond)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)  # (B,T,n_spatial,d_spatial)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B,T,n_agent,d_model)

        return x1_hat, h_t


def recon_loss_from_mae(pred_btnd: torch.Tensor,
                        target_btnd: torch.Tensor,
                        mae_mask_btNp1: torch.Tensor,
                        pixel_weights_btnp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE reconstruction over masked patches.
    pixel_weights_btnp: optional (B,T,Np,PP) per-pixel weights for rendered BEV (emphasize pedestrians, TLs).
    Broadcasts to (B,T,Np,Dp) by repeating each weight for the C channels of that pixel.
    """
    mask = mae_mask_btNp1.to(dtype=torch.float32)  # (B,T,Np,1)
    diff = (pred_btnd.float() - target_btnd.float())  # (B,T,Np,Dp)
    sq = diff.mul(diff)
    if pixel_weights_btnp is not None:
        # (B,T,Np,PP) -> (B,T,Np,Dp) by repeating per pixel for C channels
        C = diff.shape[-1] // pixel_weights_btnp.shape[-1]
        w = pixel_weights_btnp.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape_as(sq)
        sq = sq * w
    sq = sq * mask
    if pixel_weights_btnp is not None:
        w = pixel_weights_btnp.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape_as(sq)
        denom = (mask * w).sum().clamp_min(1e-6)
    else:
        denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def recon_loss_bce_from_mae(pred_btnd: torch.Tensor,
                            target_btnd: torch.Tensor,
                            mae_mask_btNp1: torch.Tensor,
                            eps: float = 1e-7) -> torch.Tensor:
    """
    BCE reconstruction loss over masked patches only. For binary BEV masks.
    pred: [0,1] (sigmoid output), target: [0,1] (binary masks).
    """
    mask = mae_mask_btNp1.to(dtype=torch.float32)  # (B,T,Np,1)
    pred = pred_btnd.float().clamp(eps, 1.0 - eps)
    target = target_btnd.float()
    # Normalize target if in [0,255]
    if target.max() > 1.5:
        target = target / 255.0
    target = target.clamp(0.0, 1.0)

    bce = F.binary_cross_entropy(pred, target, reduction="none")  # (B,T,Np,Dp)
    bce_masked = bce * mask
    denom = mask.sum().clamp_min(1.0) * bce.shape[-1]
    return bce_masked.sum() / denom


def recon_loss_ce_from_mae(
    pred_logits_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    num_classes: int,
    patch: int,
    target_is_indices: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy reconstruction over masked patches. For semantic maps.
    pred_logits_btnd: (B,T,Np, num_classes*PP) raw logits
    target_btnd: (B,T,Np, C*PP) one-hot 0/255 if not target_is_indices,
                 else (B,T,Np, PP) float in [0,1] as class_index/(num_classes-1)
    mae_mask_btNp1: (B,T,Np,1) bool, True where masked
    target_is_indices: True when target is 1-channel int labels normalized to [0,1]
    label_smoothing: softens targets (reduces overconfident wrong classes / speckle FPs).
    focal_gamma: if >0, multiply CE by (1-p_t)^gamma (down-weights easy pixels, e.g. background).
    """
    B, T, Np, Dp = pred_logits_btnd.shape
    PP = patch * patch
    assert Dp == num_classes * PP

    mask = mae_mask_btNp1.to(dtype=torch.float32)
    pred = pred_logits_btnd.float().view(B, T, Np, PP, num_classes)
    if target_is_indices:
        # Targets are class indices linearly encoded in [0,1] as index/(num_classes-1) (not /255).
        denom = float(max(num_classes - 1, 1))
        target_idx = (target_btnd.float() * denom).round().long().clamp(0, num_classes - 1)
    else:
        target_onehot = target_btnd.float()
        if target_onehot.max() > 1.5:
            target_onehot = target_onehot / 255.0
        target_onehot = target_onehot.view(B, T, Np, num_classes, PP)
        target_idx = target_onehot.argmax(dim=-2).clamp(0, num_classes - 1)

    flat_logits = pred.reshape(-1, num_classes)
    flat_tgt = target_idx.reshape(-1).long()
    ce = F.cross_entropy(
        flat_logits,
        flat_tgt,
        weight=class_weights,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
    )
    if focal_gamma > 0.0:
        # p_t ≈ exp(-ce) for hard labels; stabilizes easy-example down-weighting with smoothing.
        pt = torch.exp(-ce.clamp(max=80.0))
        ce = ((1.0 - pt).clamp(min=0.0, max=1.0) ** float(focal_gamma)) * ce
    ce = ce.view(B, T, Np, PP)
    ce_masked = ce * mask  # mask (B,T,Np,1) broadcasts over PP
    denom = mask.sum().clamp_min(1.0) * float(PP)
    return ce_masked.sum() / denom


def recon_loss_bce_logits_from_mae(
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    pos_weight_c: torch.Tensor,  # (C,) per-channel
    C: int,
    patch: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    BCEWithLogits-style reconstruction over masked patches only, with per-channel weights.
    pred_btnd: probabilities in [0,1]; converted to logits internally.
    pos_weight_c: (C,) tensor of per-channel positive weights.
    """
    mask = mae_mask_btNp1.to(dtype=torch.float32)  # (B,T,Np,1)

    B, T, Np, Dp = pred_btnd.shape
    PP = patch * patch
    assert Dp == C * PP, f"Dp={Dp} != C*patch*patch={C*PP}"

    pred = pred_btnd.float().clamp(eps, 1.0 - eps)
    logits = torch.log(pred / (1.0 - pred))  # (B,T,Np,Dp)

    target = target_btnd.float()
    if target.max() > 1.5:
        target = target / 255.0
    target = target.clamp(0.0, 1.0)

    logits_c = logits.view(B, T, Np, C, PP)
    target_c = target.view(B, T, Np, C, PP)

    # Unweighted BCE per pixel
    bce = F.binary_cross_entropy_with_logits(logits_c, target_c, reduction="none")  # (B,T,Np,C,PP)

    # Apply per-channel pos_weight manually
    w = pos_weight_c.to(device=logits.device, dtype=logits.dtype).view(1, 1, 1, C, 1)
    bce = bce * w

    # Apply MAE mask across patches
    bce_masked = bce * mask.unsqueeze(-2)  # broadcast over C,PP
    denom = mask.sum().clamp_min(1.0) * float(Dp)
    return bce_masked.sum() / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd.float(), H, W, C, patch)
    tgt   = temporal_unpatchify(target_btnd.float(),        H, W, C, patch)

    if subsample_frac < 1.0:
        B, T = recon.shape[:2]
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt   = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0).float()
    tgt   = (tgt.clamp(0, 1)   * 2.0 - 1.0).float()

    B, T = recon.shape[:2]
    recon = recon.reshape(B * T, C, H, W)
    tgt   = tgt.reshape(B * T, C, H, W)

    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(recon, tgt)
    return lp.mean()
