# Isaac Sim hosting: tonight's setup

Goal for tonight: run Isaac Sim 6.2 on a cloud GPU, see its screen live, and try the Omniverse tools: an environment and a robot, the pick-and-place example, then our own reconstructed scene.

Prices and approval times below come from public pages and forums checked on 2026-10-30. Check the AWS console before launching.

## 1. Host decision

**Primary: AWS, g6e.2xlarge, with NVIDIA's Isaac Sim Development Workstation AMI.**
**Fallback if the AWS quota is not approved in time: Vast, headless only (no live GUI).**

| | AWS (g6e.2xlarge, Isaac Sim AMI) | Vast (RTX 4090 container) |
|---|---|---|
| Isaac Sim GUI | Works. NVIDIA's AWS guide covers it, and the AMI has Amazon DCV set up | Not confirmed. No Isaac Sim template found, and forum reports show WebRTC failing in headless 4090 containers (`StreamSdkException 800b0000`) and GUI failing in unprivileged containers |
| Setup time tonight | About 30 to 45 minutes once the quota is approved (the AMI is ready to go, first launch warms up shaders for 15+ minutes) | About 30 minutes for headless runs. A live screen could take the whole evening or not work |
| Cost per hour | About $2.24 (g6e.2xlarge, approximate). g6e.xlarge about $1.86 | About $0.30 to $0.60 for a 4090, depending on the host |
| Reliability | High. Machines stay up, and the disk keeps everything after a stop | Varies by host. A machine can disappear mid-run |
| Approval wait | **Blocker:** new accounts start with 0 vCPUs for G instances. Approval has taken anywhere from 30 minutes to over a week | None. Pay and launch |

**Why AWS:** tonight is about seeing the simulation and clicking around, and the GUI is the part Vast can't promise. NVIDIA's own AWS guide recommends exactly this setup (g6e.2xlarge or g7e.8xlarge), and the Contoro project already ran Isaac Sim on AWS.

**Why g6e.2xlarge and not smaller:** Isaac Sim 6.2 needs at least 32 GB of RAM and an RTX GPU with 16 GB of VRAM (64 GB of RAM recommended). The g6e.2xlarge has 64 GiB of RAM and an L40S (48 GB). The g6e.xlarge (32 GiB) meets the minimum and is fine for light scenes. The g6.xlarge (16 GiB) is below the minimum, and the g6.2xlarge (32 GiB, L4) works but its GPU is weaker than the RTX 4080 minimum.

## 2. Before dinner: account and quota (Sunghwan, now)

1. Sign in to the AWS console, pick one region and stay in it: **us-east-2** (where our project lives), **us-east-1** or **us-west-2** (all three have g6e).
2. If the console shows a **Free plan** with an **Upgrade plan** button, upgrade to the Paid plan first. Free plan accounts can't launch GPU instances (the g6e is greyed out as "not eligible under the Free Plan"), and remaining credits still apply after the upgrade.
3. Open **Service Quotas > AWS services > Amazon EC2**. Search for **"Running On-Demand G and VT instances"**.
   - The value is counted in vCPUs. A g6e.2xlarge needs **8**. Request **16** so a second machine or a bigger one fits later.
   - Use case to paste: "Running NVIDIA Isaac Sim for robotics simulation research at the University of Wisconsin-Madison. One g6e.2xlarge, stopped when not in use."
4. If the request still says "Pending" at dinner, open the support case it created and add the same text. Small first requests are sometimes approved automatically within an hour, and larger ones go to a person.
5. Subscribe to the AMI (free, you only pay for the instance): AWS Marketplace, search **"NVIDIA Isaac Sim Development Workstation"**, choose the **Linux** version, **Continue to Subscribe**.
6. Set the budget alarm now (see section 7), so it's in place before anything runs.

If the quota isn't approved by about 8 pm, switch to the Vast fallback in section 6.

## 3. Launch the instance (tonight)

1. Marketplace subscription page > **Continue to Configuration** > **Continue to Launch** > launch through EC2.
2. Instance type: **g6e.2xlarge**.
3. Key pair: create one (`isaac-key`), download the `.pem`, then `chmod 400 isaac-key.pem`.
4. Storage: **512 GB gp3**, the AMI's minimum. It bills even while the machine is stopped (about $41 a month), so terminate it if nobody will use it for a while.
5. Security group: allow **only your own IP** (`<your-ip>/32`), never `0.0.0.0/0`, because the streaming ports have no password or encryption:
   - 22 TCP (SSH)
   - 8443 TCP (Amazon DCV)
   - 49100 TCP and 47998 UDP (WebRTC streaming, only if you use it)
6. Advanced details > **Shutdown behavior: Stop**, so the auto-stop timer in section 7 stops the machine instead of deleting it.
7. Launch, then copy the **public IP**.

Teammates on other networks need their own IP added to the security group, or one person shares their screen.

From a laptop with the AWS CLI (profile `robot`), `bash sim/isaac_aws.sh launch` does all of the above with the key `isaac-key`, the group `isaac-sim` and the DCV license role `isaac-sim-dcv`, and schedules an auto-stop 3 hours after every boot. The same script has `ip`, `stop`, `start` and `terminate`.

## 4. See the screen

**Option A, Amazon DCV (easiest, full desktop):**

```bash
ssh -i isaac-key.pem ubuntu@<public_ip>
```

```bash
sudo passwd ubuntu
```

```bash
sudo dcv list-sessions
```

A session named `console` should be listed. Then open `https://<public_ip>:8443` in a browser or the DCV client and log in as `ubuntu`.

On the desktop, start Isaac Sim natively (first time only, warmup takes 15+ minutes):

```bash
sudo chown -R ubuntu:root /opt/IsaacSim
cd ~/IsaacSim && ./post_install.sh && ./warmup.sh
```

```bash
cd ~/IsaacSim && ./isaac-sim.sh
```

**Option B, container plus WebRTC streaming client** (use this if you'd rather run the official container, `nvcr.io/nvidia/isaac-sim:6.2.0`):

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.2.0
mkdir -p ~/docker/isaac-sim/cache/{main/ov,main/warp,computecache,kit} ~/docker/isaac-sim/{config,data/documents,data/Kit,logs,pkg}
sudo chown -R 1234:1234 ~/docker/isaac-sim
```

```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  -e "PRIVACY_CONSENT=Y" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -u 1234:1234 \
  nvcr.io/nvidia/isaac-sim:6.2.0
```

Inside the container:

```bash
PUBLIC_IP=$(curl -s ifconfig.me) && ./runheadless.sh --/exts/omni.kit.livestream.app/primaryStream/publicIp=$PUBLIC_IP \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998
```

Then connect with the **Isaac Sim WebRTC Streaming Client** app (download from NVIDIA) to the public IP. Add `--runtime=nvidia` to `docker run` if the GPU is not detected. If `docker pull` is denied, sign in at ngc.nvidia.com and accept the Isaac Sim terms first.

**Isaac Lab:** not needed tonight. Every test below runs in Isaac Sim alone. Install it later, when we train policies (for example CooHOI-style carrying), following the Isaac Lab install guide for the matching Isaac Sim version.

## 5. Test list for tonight

Work through these in order and note the time each one takes.

1. **Warehouse plus a robot.** In the Content browser, open the Isaac Sim assets and load `/Isaac/Environments/Simple_Warehouse/full_warehouse.usd` (lighter options: `warehouse.usd`, `warehouse_with_forklifts.usd`, `warehouse_multiple_shelves.usd`). Drag in a robot from the Robots folder (see Seohee's robot assets page). Press Play and check that it stands and nothing falls through the floor. Assets stream from NVIDIA's cloud by default, so the first load is slow.
2. **Pick and place.** A UR10e with a Robotiq 2F-140 gripper, planned with cuMotion. From the Isaac Sim folder:

   ```bash
   ./python.sh standalone_examples/tutorials/manipulation/tutorial_9_pick_place_cumotion.py
   ```

   Success means the arm picks the object and places it at the target. Check the console for errors.
3. **SmartSpaces dataset (expectation change).** `nvidia/PhysicalAI-SmartSpaces` does **not** ship Omniverse or USD scenes. It holds multi-camera warehouse videos (1080p MP4), depth maps, calibration, ground truth and a top-down `map.png`, under CC-BY-4.0 (commercial use allowed with attribution). It's 6 TB in total, so download one scene only, without depth maps. Check the exact folder names in the Files tab first, then for example:

   ```bash
   huggingface-cli download nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
     --include "MTMC_Tracking_2025/*Warehouse_000*" --exclude "*depth*" --local-dir smartspaces
   ```

   Tonight's use: look at a camera video and its `calibration.json`, and treat it as input for our reconstruction (or NVIDIA's NuRec) later. It can't be opened as a scene directly.
4. **Our own mesh.** On a laptop with the repo, export a reconstructed scene:

   ```bash
   cd sim/sunny-scene-recon && python export_scene.py SCENE_DIR
   ```

   This writes `SCENE_DIR/export/`: `scene.urdf` (one fixed link, the textured scan as the visual, floor, walls and hulls as collisions), `ground.obj` (outdoor terrain), the visual mesh with its texture, the hull pieces, and `carry_task.json`. Copy the `export/` folder to the instance (`scp -r`). In Isaac Sim, import `scene.urdf` with the URDF importer (File > Import), keeping the base fixed, and import `ground.obj` as a static mesh with triangle-mesh collision. Drop a robot on it and press Play: it should stand on the floor and stop at the walls.

## 6. Fallback: Vast, headless

Use this only if the AWS quota isn't approved. Skip the live screen and record videos instead.

1. Rent an RTX 4090 (or better) with at least 32 GB of RAM, 100 GB of disk and a high reliability score, on the official `nvcr.io/nvidia/isaac-sim:6.2.0` image.
2. Run the pick-and-place example and our scene headless, with a camera writing frames or a video, then download the files and watch them.
3. Don't spend the evening on WebRTC there: it needs a fixed UDP port and GPU display access that Vast containers usually don't give.

## 7. Stop script and budget alarm

**Auto-stop timer.** Run this right after connecting. It stops the machine 3 hours later even if everyone forgets (needs Shutdown behavior: Stop):

```bash
sudo shutdown -h +180
```

Cancel with `sudo shutdown -c`, or run it again for more time.

**Stop from a laptop** (AWS CLI configured, same region):

```bash
aws ec2 stop-instances --instance-ids <instance-id>
```

```bash
aws ec2 describe-instances --instance-ids <instance-id> --query "Reservations[].Instances[].State.Name"
```

It should say `stopped`. A stopped machine still costs about $41 a month for its 512 GB disk. To delete it entirely, use `terminate-instances` instead (this deletes the disk too).

**Budget alarm.** In the console, open **Billing > Budgets > Create budget > Cost budget**, monthly, amount **$50**, with email alerts at 50% and 80% of actual spend and 100% of forecasted spend.

**Idle alarm (optional).** On the instance, open **Actions > Monitor > Manage CloudWatch alarms**. Add an alarm on CPUUtilization below 5% for 60 minutes, with the action **Stop this instance**.

## Sources

- [Isaac Sim AWS deployment (6.2.0)](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_advanced_cloud_setup_aws.html)
- [Isaac Sim requirements (6.2.0)](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
- [Isaac Sim environment assets (6.2.0)](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_environments.html#environments-warehouses)
- [Isaac Sim robot assets (6.2.0)](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots.html)
- [Pick-and-place tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/tutorial_pickplace_example.html)
- [PhysicalAI-SmartSpaces dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
- [Forum: WebRTC failing on a headless RTX 4090 container](https://forums.developer.nvidia.com/t/isaac-sim-6-0-0-webrtc-livestream-fails-with-streamsdkexception-800b0000-on-a-headless-rtx-4090-docker-server/372612)
- [Forum: GUI in an unprivileged container](https://forums.developer.nvidia.com/t/isaac-sim-6-0-1-gui-in-unprivileged-lxc-docker-failed-to-acquire-modesetting-permission-any-way-to-run-interactive-gui-without-drm-master-web/380016)
- [AWS re:Post: G instance quota wait times](https://repost.aws/questions/QUBBbH23A8TliNJqb8XliSYg/how-long-to-wait-for-a-response-to-a-quota-increase-reply)
