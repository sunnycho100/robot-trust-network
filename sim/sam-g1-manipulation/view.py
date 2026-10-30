"""Launch the native viewer with uv's macOS Python shared-library path resolved."""

import os
import runpy
import sys
from pathlib import Path

run_file = Path(__file__).with_name("run.py")
args = sys.argv[1:]
if args and args[0] == "agent":
    args.pop(0)
    # Qt owns the main thread and embeds MuJoCo's GL viewport in one window.
    # mjpython is only needed for the separate legacy GLFW viewer below.
    if sys.platform == "darwin":
        from desktop_launcher import launch_macos

        launch_macos(args)
    sys.argv = [str(Path(__file__).with_name("native_agent.py")), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")
    raise SystemExit
elif "--viewer" not in args:
    args.append("--viewer")
if sys.platform == "darwin":
    # uv's relocatable Python needs this when mjpython dlopens its interpreter.
    # Set it inside Python: macOS strips DYLD_* across /bin/sh trampolines.
    library_dir = str(Path(sys.base_prefix) / "lib")
    previous = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = library_dir + ":" + previous
    sys.argv = [str(Path(sys.executable).with_name("mjpython")), str(run_file), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    sys.argv = [str(run_file), *args]
    runpy.run_path(str(run_file), run_name="__main__")
