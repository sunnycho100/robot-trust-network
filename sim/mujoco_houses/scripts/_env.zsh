# Sourced by the run_* scripts: pick the demo Python and put src/ on PYTHONPATH.
repo_root="${0:A:h:h}"

if [[ -n "${UNITREE_DEMO_PYTHON:-}" ]]; then
  demo_python="$UNITREE_DEMO_PYTHON"
else
  demo_python="$(command -v python || command -v python3)"
fi
if ! "$demo_python" -c "import mujoco, numpy, torch, yaml" >/dev/null 2>&1; then
  print -u2 "$demo_python cannot import mujoco/torch/yaml."
  print -u2 "Activate the env that has requirements.txt installed, or set UNITREE_DEMO_PYTHON."
  exit 1
fi
if [[ ! -f "$repo_root/third_party/go1_policy.onnx" ]]; then
  print -u2 "Robot models are missing. Run ./scripts/fetch_assets once."
  exit 1
fi

require_broker() {
  if ! nc -z 127.0.0.1 1883 >/dev/null 2>&1; then
    print -u2 "No MQTT broker on 127.0.0.1:1883. Start it first: ./scripts/run_mqtt_broker"
    exit 1
  fi
}

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
