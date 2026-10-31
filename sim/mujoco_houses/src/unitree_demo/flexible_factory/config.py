import json
from pathlib import Path

CONFIG_PATH=Path(__file__).resolve().parents[3]/'factory/flexible_factory.json'

def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())
