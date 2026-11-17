"""Learned collision-risk head for MPPI candidate scoring.

One head on the world model's imagined rollout:

    r(s, a)  observed context latent + imagined latents h(a)
             -> per future step, per event type: P(the event has happened by step k)

The only supervision is the simulator's own recorded outcome (``safety_events``: collision,
red-light violation, stop-sign violation). No rule-derived quantity enters the head, its
targets, or its use in the planner.

Planning subtracts

    penalty(s, a) = mean_k  sum_j  lambda_j * P_j(by step k | a)

from the candidate return. Because the target is cumulative, an event predicted early weighs
more than the same event predicted late, and a candidate that avoids it entirely pays nothing.

Two properties matter and are enforced rather than hoped for:

* Action sensitivity. The head reads latents imagined under the candidate action sequence, so
  its output can differ between candidates. Measured on the previous rule-gated head, full
  throttle into a hazard scored 0.578 against 0.275 for full braking, so the mechanism works.
* Braking monotonicity. That same head also scored sustained braking riskier than cruising
  (0.093 against 0.012 on an empty road), because in logged data hard braking co-occurs with
  danger. ``monotonicity_loss`` penalises exactly that: braking harder may not raise the
  predicted risk. It needs no counterfactual outcome label, so it also applies online.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

EVENTS = ("collision", "red_light", "stop_sign")
N_EVENTS = len(EVENTS)
# CAT's MetaDrive benchmark (mixed_quality_md): vehicle contact is the event CAT's crash rate counts
MD_EVENTS = ("vehicle_contact", "object_contact", "out_of_road")
# The F1TENTH twin (mixed_quality_f1tenth): the only safety event is a wall contact, which ends the episode
F1_EVENTS = ("wall_contact",)


class RiskHead(nn.Module):
    def __init__(
        self,
        *,
        d_latent: int = 256,
        hidden: int = 256,
        dropout: float = 0.1,
        n_events: int = N_EVENTS,
        detach_inputs: bool = True,
        event_weights: Optional[Sequence[float]] = None,
        event_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.d_latent = int(d_latent)
        self.n_events = int(n_events)
        self.detach_inputs = bool(detach_inputs)
        # names only label the per-event statistics and the saved config; CARLA's by default
        self.event_names = tuple(str(x) for x in event_names) if event_names is not None else EVENTS[: self.n_events]
        if len(self.event_names) != self.n_events:
            raise ValueError(f"{len(self.event_names)} event names for {self.n_events} events")
        self.net = nn.Sequential(
            nn.LayerNorm(2 * self.d_latent),
            nn.Linear(2 * self.d_latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_events),
        )
        w = torch.zeros(self.n_events)
        if event_weights is None:
            w[0] = 1.0                      # collision only unless the planner asks for more
        else:
            w = torch.tensor([float(x) for x in event_weights], dtype=torch.float32)
        self.register_buffer("event_weights", w, persistent=True)

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def keyframe(context_latents: torch.Tensor) -> torch.Tensor:
        """(B,D), (B,C,D) or (B,C,1,D) -> the last context step as (B,D)."""
        x = context_latents
        if x.ndim == 4:
            x = x.squeeze(2)
        if x.ndim == 3:
            x = x[:, -1]
        if x.ndim != 2:
            raise ValueError(f"context_latents must be (B,D), (B,C,D) or (B,C,1,D), got {tuple(context_latents.shape)}")
        return x

    def forward(self, context_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
        """-> (B, H, n_events) logits: P(event j has happened by imagined step k)."""
        ctx = self.keyframe(context_latents)
        fut = future_latents
        if fut.ndim == 4:
            fut = fut.squeeze(2)
        if fut.ndim != 3 or fut.shape[0] != ctx.shape[0]:
            raise ValueError(f"future_latents must be (B,H,D) aligned with context, got {tuple(future_latents.shape)}")
        if self.detach_inputs:
            ctx, fut = ctx.detach(), fut.detach()
        ctx, fut = ctx.float(), fut.float()
        joint = torch.cat([ctx[:, None].expand(-1, fut.shape[1], -1), fut], dim=-1)
        return self.net(joint)

    def penalty(self, context_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
        """(B,) quantity the planner subtracts, in [0, sum(event_weights)]."""
        p = torch.sigmoid(self.forward(context_latents, future_latents))
        return (p * self.event_weights.to(p.dtype)).sum(dim=-1).mean(dim=1)

    # ------------------------------------------------------------------ training
    @staticmethod
    def targets(
        events_future: torch.Tensor,
        transition_valid_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative event targets from the recorded simulator outcomes.

        events_future            (B,H,E) bool/float: event j recorded at future step k
        transition_valid_future  (B,H)   bool: step k is a real transition (not padding)
        returns (targets (B,H,E) float, valid (B,H,E) bool). Validity is per event: after the
        event has happened its channel stays supervised through the padding, while a channel with
        no event is unsupervised there because the outcome is simply unknown.
        """
        ev = events_future.bool() & transition_valid_future.bool()[..., None]
        cum = torch.cummax(ev.long(), dim=1).values.float()
        # after the event the rollout is still supervised as "it has happened"
        valid = transition_valid_future.bool()[..., None] | (cum > 0)
        return cum, valid

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        *,
        pos_weight: float = 20.0,
    ) -> dict[str, torch.Tensor]:
        logits = logits.float()
        per = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=torch.full((), float(pos_weight), device=logits.device),
            reduction="none",
        )
        m = valid.to(per.dtype).expand_as(per)
        loss = (per * m).sum() / m.sum().clamp(min=1.0)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            stats: dict[str, torch.Tensor] = {}
            for j, name in enumerate(self.event_names):
                vj = valid[..., j] if valid.shape[-1] == targets.shape[-1] else valid[..., 0]
                tj = targets[..., j].bool() & vj
                nj = (~targets[..., j].bool()) & vj
                stats[f"{name}_pos_frac"] = tj.float().mean()
                stats[f"{name}_pred_pos"] = p[..., j][tj].mean() if bool(tj.any()) else torch.zeros((), device=p.device)
                stats[f"{name}_pred_neg"] = p[..., j][nj].mean() if bool(nj.any()) else torch.zeros((), device=p.device)
        return {"loss": loss, **stats}

    def monotonicity_loss(
        self,
        context_latents: torch.Tensor,
        future_latents: torch.Tensor,
        braked_future_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge: braking harder from the same context may not raise the predicted risk.

        Needs no outcome label, so the same term regularises offline training and online adaptation.
        """
        # Dropout must be off: two stochastic passes would differ even for identical inputs and
        # the hinge would charge that noise as a violation.
        # The hinge trains the head only. With detach_inputs=False (joint training) the BCE term is
        # allowed to shape the world model, but this term must not: its cheapest way to lower
        # relu(p_braked - p) would be to raise the executed rollout's risk through the latents.
        ctx, fut, fut_b = context_latents.detach(), future_latents.detach(), braked_future_latents.detach()
        was_training = self.training
        self.eval()
        try:
            p = torch.sigmoid(self.forward(ctx, fut))
            p_braked = torch.sigmoid(self.forward(ctx, fut_b))
        finally:
            self.train(was_training)
        return F.relu(p_braked - p).mean()

    def config_dict(self) -> dict:
        return dict(
            d_latent=self.d_latent,
            n_events=self.n_events,
            detach_inputs=self.detach_inputs,
            event_weights=[float(x) for x in self.event_weights.tolist()],
            events=list(self.event_names),
            n_params=int(sum(p.numel() for p in self.parameters())),
        )

    @classmethod
    def from_config(cls, config: dict, *, hidden: int = 256, dropout: float = 0.1) -> "RiskHead":
        """Rebuild a head from ``config_dict()`` output (as saved in checkpoints under risk_head_config)."""
        return cls(
            d_latent=int(config.get("d_latent", 256)),
            hidden=int(hidden),
            dropout=float(dropout),
            n_events=int(config.get("n_events", N_EVENTS)),
            detach_inputs=bool(config.get("detach_inputs", True)),
            event_weights=config.get("event_weights"),
            event_names=config.get("events"),
        )
