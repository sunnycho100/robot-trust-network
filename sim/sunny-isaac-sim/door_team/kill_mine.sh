#!/bin/bash
# stop my door_team renders only: the runner scripts, then the Isaac containers whose arguments name /robot/door_team/door_team.py
pkill -f "[r]un_hq.sh"; pkill -f "[r]un_v3c.sh"
for id in $(docker ps -q); do
  docker inspect --format '{{.Id}} {{.Args}}' $id | grep -q "/robot/door_team/door_team.py" && docker kill $id
done
sleep 2; docker ps --format "{{.Names}} {{.RunningFor}}"
