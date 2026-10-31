# Isaac Sim 6.2 on Brev

<!-- Usage: follow "Order of operations" in the Brev VM's Jupyter terminal. -->

Isaac Sim 6.2 in Docker on a Brev Crusoe L40S (VM Mode w/ Jupyter), seen in Chrome through NVIDIA's web viewer. Commands follow the 6.2 docs ([Brev guide](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_advanced_cloud_setup_brev.html), [web viewer](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/manual_livestream_clients.html)) and `tools/docker` at tag v6.2.0.

## Order of operations

1. In the Brev console, expose the ports below, then click **Open Notebook** and open a **Terminal**.
2. `mkdir -p ~/robot/isaac-sim`, then upload `brev_setup.sh`, `isaac_up.sh` and `docker-compose.override.yml` into that folder with the Jupyter file browser.
3. `bash ~/robot/isaac-sim/brev_setup.sh`
   - If it installs the driver, it stops: run `sudo reboot`, wait about 2 minutes, reopen the terminal and run it again.
   - If it adds you to the docker group, run `newgrp docker` and run it again.
4. Read the [NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/NVIDIA_Omniverse_License_Agreement.html). Starting the container accepts it, so `isaac_up.sh` refuses to start until you set it yourself:
   `ACCEPT_EULA=Y bash ~/robot/isaac-sim/isaac_up.sh up`
5. Wait for it to print `http://<ip>:8210` and open that in Chrome. The first start compiles shaders and can take a while.
6. Later: `bash ~/robot/isaac-sim/isaac_up.sh status`, `logs`, `logs isaac-sim` or `down`.

Only one browser tab can be connected at a time. Dozens of `OmniHub: Hub encountered error` warnings at startup are expected (no Hub cache container on a cloud VM).

## Ports to expose in Brev

| Port | Protocol | Purpose | Source |
|---|---|---|---|
| 8210 | TCP | Web viewer page | 207.110.225.211/32 |
| 49100 | TCP | WebRTC signaling | 207.110.225.211/32 |
| 47998 | UDP | WebRTC media | 207.110.225.211/32 |

The streams have no password or encryption, so never open them to everyone. Brev's Launchable settings offer "only the deployer's IP" for exposed ports, and NVIDIA's Isaac Sim Brev guide says to expose these three only to your IP. Unverified: Brev's docs don't show the port form on an existing instance's Access tab, so check there for a source IP field.

If the instance page can only open ports to all IPs, restrict them on the VM instead (the containers use host networking, so the host firewall applies). Unverified on Brev, and make sure SSH stays allowed:

```bash
sudo ufw allow 22/tcp
sudo ufw allow from 207.110.225.211 to any port 8210,49100 proto tcp
sudo ufw allow from 207.110.225.211 to any port 47998 proto udp
sudo ufw --force enable
```

## Where things live on the VM

| Path | What |
|---|---|
| `~/robot/isaac-sim/` | These scripts |
| `~/robot/` | Our demo scripts and scene files, mounted read-write at `/robot` inside Isaac Sim |
| `~/IsaacSim/tools/docker/` | NVIDIA's compose file (sparse clone of v6.2.0) |
| `~/docker/isaac-sim/`, `~/.cache/ov/hub/` | Isaac Sim caches and logs, owned by uid 1234 |

Unverified: where the Jupyter file browser opens, and whether Brev keeps `~/robot` across a stop. Brev's docs only promise `/home/ubuntu/workspace`, so keep a copy of anything important on the Mac.

## Stop or delete the machine

- **Stop** (Brev console, instance page): compute billing ends, the disk is kept and bills $0.05 an hour. Starting again can fail if Crusoe has no free L40S at that moment. The public IP can change, so run `isaac_up.sh up` again after every start.
- **Delete**: removes the machine and its disk, so all billing stops. It can't be undone.

## Demos

Live, in the web viewer: see [`demos/RUN.md`](demos/RUN.md). Each demo is its own app on the same stream, so stop the default one first.

Recorded to video (no viewer needed). `record_demo.py` runs any standalone example headless and saves its viewport as frames; the two batch scripts turn them into an mp4 and a contact sheet per demo in `~/robot/recordings/`:

```bash
bash ~/robot/isaac-sim/demos/record_all.sh 20    # stock examples: Spot, Go2, ANYmal, H2, Franka, UR10e pick and place, UR10 palletizing, Franka stacking
bash ~/robot/isaac-sim/demos/record_ours.sh      # our scanned floor (floor4_static3) imported, with an H1 walking in it
bash ~/robot/isaac-sim/demos/record_more.sh      # 60 s palletizing, the H1 at our room's center, the H1 in the NuRec wormhole scan
```

The frames are owned by the container user, so the scripts `chown` them before ffmpeg. Download the mp4s from the Jupyter file browser.

`scene_bbox.py` prints a scene's up axis and bounding box, which is how to pick a spawn point inside it.

Gated Hugging Face files (the NuRec dataset) without a token on the VM: download them in a signed-in browser, `split -b 9m` them, upload the parts with the Jupyter file browser, then `cat` them back together and check the sha256. `record_more.sh` expects the wormhole scan at `~/robot/nurec/xgrid-wormhole/wormhole.usdz`.

CooHOI (two humanoids carrying a box, Isaac Gym) has its own setup in [`coohoi/RUN.md`](coohoi/RUN.md). Each humanoid of a pair lives in its own Isaac Gym env and a camera renders only its own env, so the recorder films both envs from the same pose and merges them by depth (unverified on the VM).
