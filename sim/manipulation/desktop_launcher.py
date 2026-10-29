"""Give the Mac desktop a real app identity while using the project Python."""

import os
import plistlib
import subprocess
import sys
import sysconfig
from pathlib import Path


def launch_macos(args):
    directory = Path(__file__).resolve().parent
    contents = directory / "runs/native/Robot Lab.app/Contents"
    binary = contents / "MacOS/RobotLab"
    source = directory / "desktop_main.c"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists() or source.stat().st_mtime > binary.stat().st_mtime:
        subprocess.run(["/usr/bin/clang", str(source), "-o", str(binary)], check=True)
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleExecutable": "RobotLab",
                "CFBundleIdentifier": "org.robottrust.lab",
                "CFBundleName": "Robot Lab",
                "CFBundleDisplayName": "Robot Lab",
                "CFBundlePackageType": "APPL",
                "NSHighResolutionCapable": True,
            },
            stream,
        )
    library = Path(sys.base_prefix) / "lib" / sysconfig.get_config_var("LDLIBRARY")
    os.execv(
        str(binary),
        [
            str(binary),
            str(library),
            sys.executable,
            str(directory / "native_agent.py"),
            *args,
        ],
    )
