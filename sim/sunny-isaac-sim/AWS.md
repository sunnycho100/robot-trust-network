<!-- Usage: the Brev stack (README.md) on one AWS g6e. Run the Mac part from the repo root, then the VM parts in order. -->
# Isaac Sim on AWS

The same stack as Brev, on a g6e.2xlarge (L40S 48 GB, 64 GiB RAM) in us-east-2, from NVIDIA's Isaac Sim Development Workstation AMI (Ubuntu 24.04). It costs about $2.24 an hour, plus about $41 a month for the 512 GB disk while the machine exists. Brev's L40S is $1.79 an hour. Sunghwan approves every launch.

Already in place: key pair `isaac-key` (`~/.ssh/isaac-key.pem`), security group `isaac-sim`, DCV license role `isaac-sim-dcv`, and the Marketplace subscription. Still blocked by the G quota: 0 vCPUs, with an appeal for 8 pending in support case 179341170700218.

## On the Mac

```bash
aws login --profile robot
bash sim/sunny-isaac-sim/isaac_aws.sh open           # ports 22, 8443, 8210, 49100 and 47998/udp, this Mac's IP only
DRY=1 bash sim/sunny-isaac-sim/isaac_aws.sh launch   # checks everything, starts nothing
bash sim/sunny-isaac-sim/isaac_aws.sh launch         # starts billing
bash sim/sunny-isaac-sim/isaac_aws.sh sync           # about 2 minutes after launch, once SSH answers
bash sim/sunny-isaac-sim/isaac_aws.sh ssh
```

Run `open` again whenever the Mac's IP changes, and remove old IPs from the group in the console.

## Isaac Sim container and web viewer

The AMI's `~/IsaacSim` is NVIDIA's native install (the one its DCV desktop runs), so the compose files go in another folder. Every script here reads `REPO`.

```bash
echo 'export REPO=$HOME/isaacsim-compose' >> ~/.bashrc && export REPO=$HOME/isaacsim-compose
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
bash ~/robot/isaac-sim/brev_setup.sh
ACCEPT_EULA=Y bash ~/robot/isaac-sim/isaac_up.sh up   # Sunghwan reads the license first (README.md, step 4)
```

`brev_setup.sh` skips what the AMI already has (Docker, the NVIDIA driver). Unverified: the AMI's driver version. If `nvidia-smi` shows a driver older than 595.58.03, stop before running `brev_setup.sh`, because it would replace the driver the AMI's desktop uses.

`isaac_up.sh` prints `http://<ip>:8210`; open it in Chrome. Don't run the native Isaac Sim on the DCV desktop at the same time, since both need the GPU.

## Isaac Lab 3.0 and the CooHOI port

Follow [`coohoi_lab/README.md`](coohoi_lab/README.md) step 1, then steps 4 and 5, with these changes:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && sudo apt-get install -y ffmpeg
git clone https://github.com/Winston-Gu/CooHOI.git ~/CooHOI && git -C ~/CooHOI checkout -q 3183d99   # the checkpoints, without Isaac Gym
```

- **Torch:** uv installs a cu130 torch, which needs driver 580 or newer. Check with `nvidia-smi`. On 580 or newer, skip the cu128 line in step 1. On anything older, run it as written.
- **Isaac Gym** (`coohoi/setup_coohoi.sh`) is only for step 2's reference recording, and the play runs don't need it.
- If Kit fails to start, stop the container first: `bash ~/robot/isaac-sim/isaac_up.sh down`.

## The factory render

The factory USD only exists on the Brev VM (`~/robot/g1coop/usd/factory.usd`). Copy that folder to `~/robot/g1coop/usd/` if the Brev VM still exists. Otherwise rebuild it from Seohee's branch (steps from her `factory/SETUP.md`; unverified on this AMI's Python 3.12):

```bash
sudo apt-get install -y zsh python3-venv ffmpeg
git clone --depth 1 -b g1_cooperative_task https://github.com/sunnycho100/robot-trust-network ~/g1coop-src
cd ~/g1coop-src/sim/mujoco_houses
python3 -m venv .venv && .venv/bin/pip install -r requirements-factory.txt usd-core
./scripts/fetch_assets
git clone --filter=blob:none https://github.com/cps-test-lab/roqsim.git ../roqsim_factory_upstream
git -C ../roqsim_factory_upstream checkout 6e6999be07e56ebf442b2f20225f33b97c3a9f70
PYTHONPATH=src .venv/bin/python ~/robot/g1coop/record_mujoco.py ~/robot/g1coop/rec
.venv/bin/python ~/robot/g1coop/build_usd.py ~/robot/g1coop/rec ~/robot/g1coop/usd
```

Then render:

```bash
bash ~/robot/g1coop/run_render.sh factory_aws
```

## Stop

```bash
bash sim/sunny-isaac-sim/isaac_aws.sh stop        # compute stops, the disk keeps billing
bash sim/sunny-isaac-sim/isaac_aws.sh terminate   # deletes the machine and its disk
```

The machine also stops itself 3 hours after every boot. Run `sudo shutdown -c` on it to cancel. The public IP changes after a stop, so run `isaac_up.sh up` again after every start.
