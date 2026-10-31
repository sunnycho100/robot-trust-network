#!/bin/bash
# Colab A step 2: Isaac Sim 6.2 pip into a Python 3.12 venv. Colab's uv defaults to the system Python 3.13, so pass --python.
export UV_SYSTEM_PYTHON=false
cd /content && rm -rf isim && uv venv -p 3.12 /content/isim
time uv pip install --python /content/isim/bin/python "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match 2>&1 | tail -15
/content/isim/bin/python -c "import isaacsim, importlib.metadata as m; print('isaacsim', m.version('isaacsim'))"
du -sh /content/isim; echo PIP2_DONE
