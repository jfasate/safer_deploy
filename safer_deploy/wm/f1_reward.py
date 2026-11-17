"""Reward decomposition for the F1TENTH twin (mixed_quality_f1tenth/env.py), the counterpart of md_reward.py.

The environment's reward is

    r = track_term - C_STEER * steer_change - C_LAT * lateral_term - C_CONTACT * contact_penalty

with track_term = 1 - |ds / (v_ref * 0.1 s) - 1| (progress along the reference trajectory at the reference speed),
steer_change = |delta_t - delta_{t-1}| (normalised steering commands), lateral_term = (lateral offset / half lane
width)^2 and contact_penalty = 1 + the speed before a wall contact, 0 without one. The collector stores the four
factors in this order (``reward_factors``) and checks the reconstruction at every step.

The world model's reward-factor head (``reward_factor_kind="f1tenth"``, five raw outputs) predicts

    0  track_term                           (raw)
    1  lateral_term                         (softplus)
    2  wall-contact logit                   (sigmoid: P(contact in this step))
    3  contact penalty given a contact      (softplus: 1 + impact speed)
    4  unused (no target)

The steering change is not predicted: it is a function of the executed actions, as in CARLA's ValeoAction
reconstruction. The decomposed reward is the expectation under the predicted contact probability; the contact logit is
trained with plain (unweighted) cross-entropy so that the probability stays calibrated, the penalty magnitude only on
the steps with a contact.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

COMPONENTS = ("track_term", "steer_change", "lateral_term", "contact_penalty")
# env.EnvConfig's reward weights (checked against a collection's config by f1_dataset.F1WindowDataset)
C_STEER = 0.15
C_LAT = 0.3
C_CONTACT = 10.0


def reward_from_components(components: torch.Tensor) -> torch.Tensor:
    """The environment's reward rebuilt exactly from stored components (..., 4)."""
    c = components.float()
    return c[..., 0] - C_STEER * c[..., 1] - C_LAT * c[..., 2] - C_CONTACT * c[..., 3]


def steer_change(actions_context: torch.Tensor, actions_future: torch.Tensor) -> torch.Tensor:
    """|delta_t - delta_{t-1}| (B,H) for the future actions (B,H,2), delta_{-1} the last context action's steering
    (zero at an episode start, as the environment's last action after a reset)."""
    steer = actions_future[..., 1].float()
    previous = torch.cat([actions_context[:, -1:, 1].float(), steer[:, :-1]], dim=1)
    return (steer - previous).abs()


def expected_reward(raw: torch.Tensor, steer_changes: torch.Tensor) -> torch.Tensor:
    """Expected reward (...,) from the head's raw outputs (..., 5) and the steering changes (...,)."""
    track = raw[..., 0].float()
    lateral = F.softplus(raw[..., 1].float())
    p_contact = torch.sigmoid(raw[..., 2].float())
    penalty = F.softplus(raw[..., 3].float())
    return track - C_STEER * steer_changes.float() - C_LAT * lateral - C_CONTACT * p_contact * penalty


def factor_loss(raw: torch.Tensor, components: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Smooth-L1 on the track and lateral terms, cross-entropy of the contact and smooth-L1 on the penalty of the steps
    with a contact, over valid steps."""
    valid_f = valid.float()
    denom = valid_f.sum().clamp(min=1.0)
    c = components.float()
    contact = (c[..., 3] > 0.0).float()
    track_per = F.smooth_l1_loss(raw[..., 0].float(), c[..., 0], reduction="none")
    lateral_per = F.smooth_l1_loss(F.softplus(raw[..., 1].float()), c[..., 2], reduction="none")
    contact_per = F.binary_cross_entropy_with_logits(raw[..., 2].float(), contact, reduction="none")
    penalty_per = F.smooth_l1_loss(F.softplus(raw[..., 3].float()), c[..., 3], reduction="none")
    hit_f = contact * valid_f
    track_loss = (track_per * valid_f).sum() / denom
    lateral_loss = (lateral_per * valid_f).sum() / denom
    contact_loss = (contact_per * valid_f).sum() / denom
    penalty_loss = (penalty_per * hit_f).sum() / hit_f.sum().clamp(min=1.0)
    with torch.no_grad():
        track_mae = ((raw[..., 0].float() - c[..., 0]).abs() * valid_f).sum() / denom
    return {"loss": track_loss + lateral_loss + contact_loss + penalty_loss, "track_loss": track_loss,
            "lateral_loss": lateral_loss, "contact_loss": contact_loss, "penalty_loss": penalty_loss,
            "track_mae": track_mae}
