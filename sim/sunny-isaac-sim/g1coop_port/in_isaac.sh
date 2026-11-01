#!/bin/bash
# Run a python script in the Isaac Sim 6.2 container (headless). Usage (on the VM): bash in_isaac.sh LOG script.py args...
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
C="docker compose -p isim -f $HOME/IsaacSim/tools/docker/docker-compose.yml -f $HOME/robot/isaac-sim/docker-compose.override.yml"
log=$1; shift
sudo setfacl -R -m u:1234:rwX $HOME/robot/g1coop
$C run --rm --entrypoint bash isaac-sim -c "cd /isaac-sim && ./python.sh $*" > $log 2>&1
echo EXIT $?
