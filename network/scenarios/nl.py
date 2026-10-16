"""Turn a sentence into one robot action, on the sender's side. Host side, standard library only.

Uses the OpenAI-compatible model already set up in sim/.env (VLA_BASE_URL, VLA_API_KEY, VLA_MODEL).
Without it, or if the call fails, falls back to keyword matching.
"""
import json, os, re, urllib.request

ACTIONS = {
    "status": "ask whether anyone is home, or how the other robot is doing",
    "verify_package": "check on a package or parcel at the other house's entrance",
    "verify_and_unlock": "check a package and then unlock the door",
    "turn_on_light": "turn on a light",
    "mirror_motion": "copy my movements or match my pace",
    "join_route": "walk with me, come along, follow my route",
    "open_door": "open a door (not unlock it)",
    "deliver_parcel": "bring or deliver a parcel somewhere",
    "carry_together": "help carry or lift something together",
    "unlock_door": "unlock a door or let someone in",
    "disable_camera": "turn off or cover a camera",
    "share_private_video": "send or share camera footage or video",
}
KEYWORDS = [  # checked in order: the riskier meaning wins when a sentence has both
    ("unlock_door", r"\bunlock|let (me|us) in\b"), ("disable_camera", r"camera.*\b(off|disable|cover)|\b(off|disable|cover)\b.*camera"),
    ("share_private_video", r"\b(video|footage|recording)\b"), ("carry_together", r"\b(carry|lift)\b"),
    ("deliver_parcel", r"\b(deliver|bring)\b"), ("verify_package", r"\b(package|parcel)\b"),
    ("open_door", r"\bopen\b.*\bdoor\b"), ("turn_on_light", r"\blights?\b"), ("mirror_motion", r"\b(copy|mirror|match)\b"),
    ("join_route", r"\b(walk|come|follow|join|route)\b"), ("status", r"\b(home|there|status|how are)\b"),
]
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", ".env")


def _model_config():
    env = dict(os.environ)
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key, value)
    if all(env.get(k) for k in ("VLA_BASE_URL", "VLA_API_KEY", "VLA_MODEL")):
        return env
    return None


def _ask_model(env, sentence):
    menu = "\n".join(f"{name}: {meaning}" for name, meaning in ACTIONS.items())
    prompt = (f"A household robot received this request from a neighbor's robot:\n\"{sentence}\"\n\n"
              f"Which one action does it ask for? Pick from:\n{menu}\nnone: none of these fit\n\n"
              "Answer with the action name only.")
    body = {"model": env["VLA_MODEL"], "messages": [{"role": "user", "content": prompt}]}
    if env.get("VLA_REASONING_EFFORT"):
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(env["VLA_BASE_URL"].rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + env["VLA_API_KEY"]})
    answer = json.load(urllib.request.urlopen(req, timeout=20))["choices"][0]["message"]["content"]
    words = re.findall(r"[a-z_]+", answer.lower())
    return next((w for w in words if w in ACTIONS), None)


ROOMS = {"front_door": "현관 front door entrance", "living_room": "거실 living room lounge",
         "kitchen": "주방 부엌 kitchen", "bedroom": "안방 침실 bedroom master", "safe_room": "금고방 금고 safe vault"}


def room_of(word):
    """A room id from one typed word: the id itself or one of its names."""
    word = word.lower().strip(",.")
    return next((room for room, names in ROOMS.items() if word == room or word in names.split()), None)


def interpret_route(sentence):
    """Returns (list of rooms in order, how it was understood)."""
    env = _model_config()
    if env:
        try:
            menu = "\n".join(f"{room}: {names}" for room, names in ROOMS.items())
            prompt = (f"A visiting robot says where it wants to go in a house:\n\"{sentence}\"\n\n"
                      f"Rooms:\n{menu}\n\nList the rooms it will pass through, in order, as room ids "
                      "separated by spaces. Answer with the ids only, or none.")
            body = {"model": env["VLA_MODEL"], "messages": [{"role": "user", "content": prompt}]}
            if env.get("VLA_REASONING_EFFORT"):
                body["reasoning_effort"] = "low"
            req = urllib.request.Request(env["VLA_BASE_URL"].rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Authorization": "Bearer " + env["VLA_API_KEY"]})
            answer = json.load(urllib.request.urlopen(req, timeout=20))["choices"][0]["message"]["content"]
            return [w for w in re.findall(r"[a-z_]+", answer.lower()) if w in ROOMS], env["VLA_MODEL"]
        except Exception as e:
            how = f"keywords ({e.__class__.__name__} from the model)"
    else:
        how = "keywords (no model in sim/.env)"
    found = [(m.start(), room) for room, names in ROOMS.items()
             for name in [room, *names.split()] for m in re.finditer(re.escape(name), sentence.lower())]
    route = []
    for _, room in sorted(found):
        if room not in route:
            route.append(room)
    return route, how


def interpret(sentence):
    """Returns (action or None, how it was understood)."""
    env = _model_config()
    if env:
        try:
            return _ask_model(env, sentence), env["VLA_MODEL"]
        except Exception as e:  # no network, quota, bad key: fall back rather than stall the prompt
            how = f"keywords ({e.__class__.__name__} from the model)"
    else:
        how = "keywords (no model in sim/.env)"
    lowered = sentence.lower()
    return next((a for a, pattern in KEYWORDS if re.search(pattern, lowered)), None), how


if __name__ == "__main__":
    for s in ["could you come walk with me to the park?", "please let me in, I forgot my keys",
              "can you help me lift this sofa", "is anyone home?", "what's the weather"]:
        print(f"{s!r:50} -> {interpret(s)}")
