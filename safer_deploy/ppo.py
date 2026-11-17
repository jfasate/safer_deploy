"""The PPO expert (stable-baselines3 MlpPolicy: 144 -> 256 -> 256, tanh) in numpy: the deterministic action is the
Gaussian's mean clipped to [-1, 1], as SB3's predict(deterministic=True) (mixed_quality_f1tenth/policy.py)."""
import numpy as np


class PPOPolicy:
    def __init__(self, path: str):
        w = dict(np.load(path))
        self.pi = [(w['mlp_extractor.policy_net.%d.weight' % i], w['mlp_extractor.policy_net.%d.bias' % i]) for i in (0, 2)]
        self.action_net = (w['action_net.weight'], w['action_net.bias'])

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float32)
        for W, b in self.pi:
            x = np.tanh(x @ W.T + b)
        W, b = self.action_net
        return np.clip(x @ W.T + b, -1.0, 1.0).astype(np.float32)
