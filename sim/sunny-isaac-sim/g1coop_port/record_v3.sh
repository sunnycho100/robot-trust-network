#!/bin/bash
# Record the v3 plant (45 s of MuJoCo, all body poses at 30 fps), check it, build the USD, fetch textures, write the look layer.
# Usage (on the host, from the work folder): bash record_v3.sh   Env: VENV, REC (recording dir, about 1.2 GB; default rec_v3)
set -e
cd "$(dirname "$0")"
PY=${VENV:-$PWD/venv}/bin/python; REC=${REC:-$PWD/rec_v3}
PYTHONPATH=code/mujoco_houses/src $PY record_mujoco.py "$REC" --duration 45 --layout realistic_layout_v3.json | grep -v '^ \|^{\|^}'
$PY -c "
import json; r = json.load(open('$REC/report.json'))
for k, v in r['carries'].items(): print(k, v['scenario'], v['state'], 'pass', v['pass_probe'], 'pen', round(v['max_penetration_m'], 4))
print('dual arm', r['dual_arm']['pass_cooperation'], 'qc', r['qc']['state'], 'spots', [d['pass_walking'] for d in r['quadrupeds'].values()])"
PYTHONPATH=code/mujoco_houses/src $PY check_penetration.py "$REC"
$PY build_usd.py "$REC" usd_v3 2>&1 | grep -v Warning | tail -1
[ -f tex_cc0/picked.json ] || $PY fetch_tex.py
$PY look_hq.py usd_v3 2>&1 | grep -v Warning | tail -1
