"""CooHOI's rl_games actor as a plain torch module (no rl_games). Usage:
    pi = CoohoiActor.load("TwoAgent.pth"); act = pi(obs)   # obs (N, 299) -> actions (N, 28) in [-1, 1], MJCF order
Same path as rl_games 1.1.4 at play time (players.py get_action, deterministic): RunningMeanStd normalisation
((obs - mean) / sqrt(var + 1e-5), clamped to +-5, stats cast to float32), the actor MLP (Linear + ReLU per layer,
sizes read from the checkpoint: 1024, 512), the mu head with no activation, then clamp to +-1. The value head, sigma,
the AMP discriminator and its normaliser are ignored.
"""
import torch
from torch import nn


class CoohoiActor(nn.Module):
    def __init__(self, sizes, obs_mean, obs_var):
        super().__init__()
        layers = []
        for a, b in zip(sizes[:-2], sizes[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.mlp, self.mu = nn.Sequential(*layers), nn.Linear(sizes[-2], sizes[-1])
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_var", obs_var.float())

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m, rms = ck["model"], ck["running_mean_std"]
        ws = [m[f"a2c_network.actor_mlp.{i}.weight"] for i in range(0, 100, 2) if f"a2c_network.actor_mlp.{i}.weight" in m]
        sizes = [ws[0].shape[1]] + [w.shape[0] for w in ws] + [m["a2c_network.mu.weight"].shape[0]]
        pi = cls(sizes, rms["running_mean"], rms["running_var"])
        sd = {f"mlp.{2 * k}.{p}": m[f"a2c_network.actor_mlp.{2 * k}.{p}"] for k in range(len(ws)) for p in ("weight", "bias")}
        sd.update({f"mu.{p}": m[f"a2c_network.mu.{p}"] for p in ("weight", "bias")})
        pi.load_state_dict({**sd, "obs_mean": pi.obs_mean, "obs_var": pi.obs_var})
        return pi.to(device).eval()

    @torch.no_grad()
    def forward(self, obs):
        x = ((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)).clamp(-5.0, 5.0)
        return self.mu(self.mlp(x)).clamp(-1.0, 1.0)
