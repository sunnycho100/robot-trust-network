#!/bin/sh
# The guided demo in the browser: starts Docker if needed, opens the house map with Start and Pause,
# and plays the story step by step in the events panel. Ctrl+C here closes the page server.
# (For the same story in the terminal instead: python3 play.py --demo)
cd "$(dirname "$0")" && exec python3 story.py "$@"
