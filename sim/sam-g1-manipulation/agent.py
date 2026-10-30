"""The robot's language brain: look, choose one action, see the result, repeat.

Gemini sees the robot's head camera (and a room overview), chooses one skill per
turn and reads a plain-language result. The skills do the walking, balancing,
reaching, grasping and placing. The user's direct instructions always win.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future

from google_brain import public_parts
from scene_memory import memory_for

MAX_STEPS = 20
TIME_BUDGET_S = 600


def tool(name, description, properties, required):
    properties = {
        **properties,
        "say": {
            "type": "STRING",
            "description": "One short, plain sentence for the person watching: what you are doing and why.",
        },
    }
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "OBJECT", "properties": properties, "required": [*required, "say"]},
    }


N = {"type": "NUMBER"}
S = {"type": "STRING"}
SIDE = {"type": "STRING", "enum": ["auto", "left", "right"]}
HAND = {"type": "STRING", "enum": ["left", "right"]}
BOX = {
    "type": "ARRAY",
    "items": N,
    "description": "[ymin, xmin, ymax, xmax] normalized 0..1000, tight around ONE object in that image",
}
POINT = {"type": "ARRAY", "items": N, "description": "[y, x] normalized 0..1000 on a visible flat surface"}
FRAME = {"type": "STRING", "description": "frame_id of the image the box/point was drawn on"}
RELATION = {"type": "STRING", "enum": ["on_top", "next_to", "at_point", "back_where_it_was"]}
CAMERAS = ["head", "overview", "front", "left_wrist", "right_wrist"]

TOOLS = [
    tool(
        "observe",
        "Get a fresh image. 'head' is the robot's own eyes (looks down in front of it); "
        "'overview' shows the whole area around the robot; wrist cameras look along a hand.",
        {"camera": {"type": "STRING", "enum": CAMERAS}},
        ["camera"],
    ),
    tool(
        "move_object",
        "Pick up one object and put it somewhere, in one go: walks, crouches/leans, grasps, "
        "carries and sets it down by feel. on_top/next_to need reference_box; at_point needs point; "
        "back_where_it_was uses memory_id from the robot's memory of earlier moves.",
        {
            "frame_id": FRAME,
            "object_box": BOX,
            "object_name": S,
            "relation": RELATION,
            "reference_box": BOX,
            "point": POINT,
            "memory_id": S,
            "side": SIDE,
        },
        ["frame_id", "object_box", "object_name", "relation"],
    ),
    tool(
        "pick_up",
        "Walk to one object and pick it up with one hand (fingers open about 8 cm).",
        {"frame_id": FRAME, "object_box": BOX, "object_name": S, "side": SIDE},
        ["frame_id", "object_box", "object_name"],
    ),
    tool(
        "place",
        "Set down the object the robot is holding: on_top of / next_to a reference object, "
        "at_point on a surface, or back_where_it_was. Lowers until it feels contact, then lets go.",
        {
            "relation": RELATION,
            "frame_id": FRAME,
            "reference_box": BOX,
            "point": POINT,
            "memory_id": S,
            "side": SIDE,
        },
        ["relation"],
    ),
    tool(
        "release",
        "Open the gripper right now and let go of whatever it holds (drop it). Always allowed.",
        {"side": SIDE},
        [],
    ),
    tool(
        "grip",
        "Set one gripper: closure 0 = fully open, 1 = closed.",
        {"side": HAND, "closure": N},
        ["side", "closure"],
    ),
    tool(
        "go_to",
        "Walk to stand facing an object (box) or a floor/surface spot (point), distance_m away (default 0.6).",
        {"frame_id": FRAME, "box": BOX, "point": POINT, "distance_m": N},
        ["frame_id"],
    ),
    tool("turn", "Turn in place; positive degrees turn left.", {"degrees": N}, ["degrees"]),
    tool(
        "walk",
        "Walk a short distance relative to the robot: forward_m (up to 3, negative backs up), left_m (up to 1).",
        {"forward_m": N, "left_m": N},
        ["forward_m"],
    ),
    tool(
        "set_posture",
        "Crouch or stand (pelvis height 0.45-0.80 m, 0.74 normal) and lean the torso forward (0-0.8 rad).",
        {"height_m": N, "torso_pitch_rad": N},
        ["height_m"],
    ),
    tool(
        "move_hand",
        "Nudge one hand relative to the robot: forward_m, left_m, up_m, each at most 0.15 m.",
        {"side": HAND, "forward_m": N, "left_m": N, "up_m": N},
        ["side"],
    ),
    tool(
        "finish",
        "End the task with an honest status: done, blocked (cannot be done by this robot) or failed.",
        {"status": {"type": "STRING", "enum": ["done", "blocked", "failed"]}, "summary": S},
        ["status", "summary"],
    ),
]

SYSTEM = """You are the brain of a Unitree G1 humanoid robot in a MuJoCo simulation.
You see through cameras and act by calling exactly one tool per turn, then you get
a measured result and a fresh image. The robot's skills handle the body for you:
turning and walking forward to a good spot, balancing, crouching and leaning,
moving the arms, grasping with two-finger grippers (they open about 8 cm), carrying,
and setting objects down gently by feel.

Pointing at things in images:
- Boxes are [ymin, xmin, ymax, xmax], points are [y, x], both normalized 0..1000,
  in the image whose frame_id you pass (use the newest image).
- Draw a box tightly around the ONE object meant. In a stack, box only the block meant.
- Box an object in an image where it is fully visible. If it is cut off at the edge
  or tiny, use the other image (head or overview) or observe again first.

Choosing actions:
- "Put/move/place X on Y" -> move_object with relation on_top.
  "next to / beside / near Y" -> relation next_to.
  "on the table / over there" -> relation at_point with a point on that surface.
  "put it back where it was" -> relation back_where_it_was with the memory_id.
- If an object is already held, use place to put it down.
- "Drop it", "let go", "open the gripper/hand" -> call release immediately.
  The user's direct instructions always come first. Never refuse to release.
- If something is far away or not in view, look with observe("overview"), or go_to / turn.
- When a skill fails, read its reason and suggestion and change something: a clearer
  view, the other hand, a different spot, going closer. Do not repeat an identical
  failing call more than twice.
- Finish with status done when the request is satisfied (check the image), blocked
  when this robot cannot do it, failed after reasonable attempts. Be specific.
- Keep "say" short and plain; it is shown to the person watching.
"""


def capture(console, epoch, camera="head"):
    future = Future()
    console.commands.put(("visual_capture", epoch, (future, camera)))
    return future.result(timeout=45)


def act(console, epoch, name, args, timeout=420):
    future = Future()
    console.commands.put(("action", epoch, (name, args, future)))
    return future.result(timeout=timeout)


def image_parts(console, observation):
    header = {
        "frame_id": observation["frame_id"],
        "camera": observation["camera"],
        "robot": observation.get("robot_status"),
    }
    return [{"text": json.dumps(header)}, console.brain.image_part(observation["image"])]


def status_text(console):
    held = getattr(console, "held", {})
    holding = {side: row["label"] for side, row in held.items()}
    return {
        "holding": holding or "nothing",
        "memory_of_moves": memory_for(console).context(),
    }


def first_call_only(content):
    """Keep the model turn up to its first function call (one action per turn)."""
    parts = []
    for part in content["parts"]:
        parts.append(part)
        if "functionCall" in part:
            break
    return {**content, "parts": parts}


def brief(value, depth=0):
    """A small JSON-safe copy of a skill result for the model: no big arrays."""
    if isinstance(value, dict):
        return {
            k: brief(v, depth + 1)
            for k, v in value.items()
            if depth < 3 and k not in ("geometry", "arm_solution", "stance")
        }
    if isinstance(value, (list, tuple)):
        return [brief(v, depth + 1) for v in value[:6]] if len(value) <= 6 else f"[{len(value)} values]"
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def compact(contents, keep_images=2):
    """Drop older images (user parts only) so requests stay small and fast."""
    seen = 0
    trimmed = []
    for message in reversed(contents):
        if message["role"] == "user":
            parts = []
            for part in message["parts"]:
                if "inlineData" in part:
                    if seen >= keep_images * 2:
                        parts.append({"text": "[older image omitted]"})
                        continue
                    seen += 1
                parts.append(part)
            message = {**message, "parts": parts}
        trimmed.append(message)
    return list(reversed(trimmed))


def run(console, request, epoch):
    started = time.monotonic()
    trial = console.output / f"actions-{time.time_ns()}"
    trial.mkdir(parents=True)
    console.last_action_trial = trial
    console.trace_epoch = epoch
    console.current_request = request
    (trial / "implementation.json").write_text(json.dumps(getattr(console, "implementation", {}), indent=2))
    console.brain.usage_path = trial / "api-usage.jsonl"
    console.message("request", request, details={"trial": str(trial)})
    head = capture(console, epoch, "head")
    overview = capture(console, epoch, "overview")
    if epoch != console.epoch:
        return
    images = []

    def remember_image(observation):
        index = len(images)
        (trial / f"{index:02d}-{observation['camera']}.jpg").write_bytes(observation["image"])
        images.append(
            {
                "frame_id": observation["frame_id"],
                "camera": observation["camera"],
                "sha256": hashlib.sha256(observation["image"]).hexdigest(),
            }
        )

    for observation in (head, overview):
        remember_image(observation)
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "User request: " + request},
                {"text": "Robot status: " + json.dumps(status_text(console))},
                *image_parts(console, head),
                *image_parts(console, overview),
            ],
        }
    ]
    outcome = "stopped: step budget used"
    failures = {}
    for step in range(MAX_STEPS):
        if epoch != console.epoch:
            return
        if time.monotonic() - started > TIME_BUDGET_S:
            outcome = "stopped: time budget used"
            break
        console.status = f"Thinking · step {step + 1}"
        decision_started = time.monotonic()
        response = console.brain.generate(compact(contents), SYSTEM, TOOLS, force_tool=True)
        if epoch != console.epoch:
            return
        _, calls = public_parts(response)
        if not calls:
            outcome = "stopped: the model returned no action"
            break
        call = calls[0]
        name, args = call["name"], dict(call.get("args", {}))
        say = args.pop("say", "")
        record = {
            "step": step,
            "tool": name,
            "args": args,
            "say": say,
            "latency_s": round(time.monotonic() - decision_started, 2),
            "usage": response.get("usageMetadata", {}),
            "model": response.get("modelVersion"),
        }
        with (trial / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        console.message("decision", say or f"Calling {name}", details=record)
        model_turn = first_call_only(response["candidates"][0]["content"])
        if name == "finish":
            status = args.get("status", "failed")
            summary = args.get("summary", "")
            outcome = f"{status}: {summary}"
            console.message("assistant", f"{'Done' if status == 'done' else status.capitalize()}: {summary}")
            break
        console.status = f"Doing · {name.replace('_', ' ')}"
        if name == "observe":
            camera = args.get("camera", "head")
            observation = capture(console, epoch, camera)
            result = {"ok": True, "summary": f"New {camera} image (frame {observation['frame_id']})"}
            extra = [observation]
        else:
            result = act(console, epoch, name, args)
            extra = []
        if epoch != console.epoch:
            return
        console.message("result", result.get("summary", ""), details=result, ok=bool(result.get("ok")))
        key = json.dumps([name, args], sort_keys=True)
        if not result.get("ok"):
            failures[key] = failures.get(key, 0) + 1
        fresh = extra or [capture(console, epoch, "head")]
        if not extra and step % 3 == 2:
            fresh.append(capture(console, epoch, "overview"))
        for observation in fresh:
            remember_image(observation)
        reply = {"name": name, "response": brief(result)}
        if "id" in call:
            reply["id"] = call["id"]
        parts = [{"functionResponse": reply}, {"text": "Robot status: " + json.dumps(status_text(console))}]
        if failures.get(key, 0) >= 2:
            parts.append({"text": "That exact call has now failed twice. Change the approach or finish."})
        for observation in fresh:
            parts += image_parts(console, observation)
        contents += [model_turn, {"role": "user", "parts": parts}]
        if console.fault:
            outcome = "stopped: simulation fault " + str(console.fault)
            break
    (trial / "images.json").write_text(json.dumps(images, indent=2))
    (trial / "result.json").write_text(
        json.dumps({"request": request, "outcome": outcome, "held": status_text(console)}, indent=2)
    )
    if not outcome.startswith(("done", "blocked", "failed")):
        console.message("assistant", "Stopped: " + outcome.split(": ", 1)[-1])
