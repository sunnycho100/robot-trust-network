#!/bin/sh
# Talk to robot B and robot C yourself, as robot A, robot D or a stranger. Type help inside.
# Starts the households if they are not running. Add --fresh to reset keys and trust.
cd "$(dirname "$0")" && exec python3 play.py "$@"
