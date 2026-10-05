"""Headless check: skills move the robots, and the model loop runs tool calls. No API key needed."""
import json, math
from types import SimpleNamespace as NS
import numpy as np
from vla_sim import Panda, Go2, ask_model


def test_panda_stack():
    r = Panda(*Panda.make())
    print(r.run("pick", {"object": "red_block"}))
    print(r.run("place_on", {"target": "blue_block"}))
    red, blue = r.pos("red_block"), r.pos("blue_block")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.02 and abs(red[2] - blue[2] - 0.04) < 0.01, (red, blue)
    assert r.run("place_at", {"x": 0.5, "y": 0}).startswith("error")
    assert "out of reach" in r.run("move_to", {"x": 3, "y": 0, "z": 0.1})
    r.image_png()


def test_go2_walk():
    r = Go2(*Go2.make())
    print(r.run("walk_to", {"target": "red_marker"}))
    assert abs(math.hypot(2.0 - r.x, 1.5 - r.y) - 0.55) < 1e-6
    r.run("sit", {}); assert r.run("walk_to_xy", {"x": 0, "y": 0}).startswith("error")
    r.run("stand", {}); r.run("turn", {"degrees": 90})


def test_loop_with_fake_model():
    calls = iter([[("pick", {"object": "green_block"})], [("done", {"summary": "ok"})]])
    def create(**kw):
        tc = [NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(next(calls))]
        msg = NS(content=None, tool_calls=tc, model_dump=lambda **_: {"role": "assistant", "tool_calls": []})
        return NS(choices=[NS(message=msg)], usage=None)
    r = Panda(*Panda.make())
    ask_model(r, "pick up green", NS(chat=NS(completions=NS(create=create))), "fake")
    assert r.held == "green_block"


if __name__ == "__main__":
    test_panda_stack(); test_go2_walk(); test_loop_with_fake_model(); print("all passed")
