#!/bin/bash
# Start the broker with the dashboard config and open the link monitor.
# Stop it with Ctrl+C.
cd "$(dirname "$0")"

if lsof -nP -iTCP:1883 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 1883 is already in use. Stop the other broker first (Ctrl+C in its tab)."
  exit 1
fi

open index.html
exec /opt/homebrew/sbin/mosquitto -c mosquitto.conf
