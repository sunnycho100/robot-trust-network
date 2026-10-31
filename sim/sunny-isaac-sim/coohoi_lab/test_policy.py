"""python test_policy.py: loads TwoAgent.pth and SingleAgent.pth into policy.CoohoiActor on CPU and checks a forward
pass on random 299-dim input against a separate float64 computation straight from the checkpoint tensors."""
import os

import torch

from joint_map import coohoi_root
from policy import CoohoiActor

torch.manual_seed(0)
outs = {}
for name in ("TwoAgent.pth", "SingleAgent.pth"):
    path = os.path.join(coohoi_root(), "coohoi/data/models", name)
    pi = CoohoiActor.load(path)
    sizes = [pi.mlp[0].in_features] + [m.out_features for m in pi.mlp if isinstance(m, torch.nn.Linear)] + [pi.mu.out_features]
    assert sizes == [299, 1024, 512, 28], sizes
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m, rms = ck["model"], ck["running_mean_std"]
    std = rms["running_var"].float().add(1e-5).sqrt()
    obs = rms["running_mean"].float() + std * torch.randn(256, 299)  # in-distribution random input
    act = pi(obs)
    assert act.shape == (256, 28) and torch.isfinite(act).all() and act.abs().max() <= 1.0

    # reference: float64, written out from the raw rl_games keys
    x = ((obs - rms["running_mean"].float()) / std).clamp(-5, 5).double()
    lin = lambda k, h: h @ m[f"a2c_network.{k}.weight"].double().T + m[f"a2c_network.{k}.bias"].double()  # noqa: E731
    want = lin("mu", lin("actor_mlp.2", lin("actor_mlp.0", x).relu()).relu()).clamp(-1, 1)
    err = (act.double() - want).abs().max().item()
    assert err < 1e-4, err
    sat = (act.abs() > 0.999).float().mean().item()
    assert sat < 0.5, f"{name}: {sat:.0%} of actions saturated on in-distribution input"
    outs[name] = act
    print(f"{name}: sizes {sizes}, |diff to float64 reference| {err:.1e}, saturated {sat:.0%}")
assert (outs["TwoAgent.pth"] - outs["SingleAgent.pth"]).abs().max() > 1e-2, "the two checkpoints should differ"
print("policy ok")
