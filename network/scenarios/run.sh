#!/bin/sh
# Build the three households and the broker in Docker, run every scenario, print the table.
# Exits 0 only if all scenarios pass. Add --keep to leave the containers running.
cd "$(dirname "$0")" && exec python3 run.py "$@"
