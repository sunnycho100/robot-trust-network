<!-- CooHOI two-agent box carry on the Brev L40S VM, recorded to an mp4. Usage: follow the steps in order. -->
# CooHOI two-agent carry on Brev

Runs the paper's `TwoAgent.pth` checkpoint with no display and saves `coohoi_twoagent.mp4` from an offscreen Isaac Gym camera that follows the box. It needs no X server and no VNC, and it does not touch the Isaac Sim container (48 GB is plenty for both).

## Order on the VM

0. On the Mac, download `IsaacGym_Preview_4_Package.tar.gz` from https://developer.nvidia.com/isaac-gym (needs your NVIDIA login and the license accept).

1. If you are also setting up Isaac Sim, run `../brev_setup.sh` first. It may upgrade the driver and ask for a reboot.

2. Upload the tarball and this folder (`setup_coohoi.sh`, `run_coohoi.sh`, `record_coohoi.py`) to the VM, through the Jupyter file browser or scp. The tarball must end up in `$HOME`:
   ```bash
   mv ~/workspace/IsaacGym_Preview_4_Package.tar.gz ~/   # only if Jupyter put it in ~/workspace
   # unverified: Brev adds an ssh alias per instance, so from the Mac: scp -r coohoi IsaacGym_Preview_4_Package.tar.gz <instance>:~
   ```

3. Open a Jupyter terminal, `cd` to this folder and check the machine:
   ```bash
   lsb_release -ds; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
   ```

4. Set up (conda env, torch 2.2.2+cu121, Isaac Gym, CooHOI at `3183d99`). Safe to rerun.
   ```bash
   bash setup_coohoi.sh
   ```
   The last lines should show `torch 2.2.2+cu121 | numpy 1.23.5 | cuda True NVIDIA L40S` and a `vulkaninfo` line naming the L40S. A WARNING there means step 5 will most likely fail; see below.

5. Record 20 simulated seconds (one full episode):
   ```bash
   bash run_coohoi.sh 20                       # writes ~/workspace/coohoi_twoagent.mp4
   OUT=~/workspace/long.mp4 bash run_coohoi.sh 60
   ```
   Look for `[record] first frame mean RGB ...` (well above 0) and then `[record] saved ...`.

6. Watch it in a notebook cell, or download it from the file browser:
   ```python
   from IPython.display import Video
   Video('/home/ubuntu/workspace/coohoi_twoagent.mp4', embed=True, width=960)
   ```

The video shows envs 0 and 1: both humanoids walk to the box, lift it together and carry it. An Isaac Gym camera renders only its own env, and each humanoid of a pair lives in its own env, so `record_coohoi.py` puts a second camera in the partner env at the same world pose and merges the two frames per pixel by depth (`composite()`, checked by `python test_record_coohoi.py`). Look for `[record] env origins ... -> global frames` (or `env-local`) on the first frame; if only one humanoid shows, rerun with `--partner_shift env` or `--partner_shift none` to force the other choice. `--one_cam` goes back to the single-camera video. Unverified until it runs on the VM: two cameras in two envs headless, `get_env_origin`, and `set_rigid_body_segmentation_id` (used to drop the partner env's still ghost box). CooHOI's green target outline is drawn only in its viewer, so it is not in the video. `--cam_env 3` in `run_coohoi.sh` films the next pair; camera distance, size and field of view are constants at the top of `record_coohoi.py`.

Brev's docs only promise that `/home/ubuntu/workspace` survives a stop. If the instance was stopped, rerun step 4.

## If it fails

- **`create_camera_sensor returned -1`, or a segfault in `create_sim`**: Isaac Gym's Vulkan cannot reach the GPU. `vulkaninfo --summary` must list the L40S, which needs the driver's graphics libs (`nvidia_icd.json` under `/usr/share/vulkan/icd.d` or `/etc/vulkan/icd.d`). Isaac Sim's renderer needs the same libs.
- **First frame mean RGB near 0 (black video)**: check whether Isaac Gym itself renders offscreen, apart from CooHOI:
  ```bash
  cd ~/isaacgym/python/examples && python graphics.py --headless   # Ctrl-C once it has written a few images
  # unverified: graphics.py writes its camera images to disk when run with --headless
  ```
- **`ImportError: libpython3.8.so.1.0`**: run `conda activate coohoi` first. Its activate hook sets `LD_LIBRARY_PATH`.
- **gymtorch build error**: `rm -rf ~/.cache/torch_extensions` and run again. It needs g++ and ninja, which step 4 installs.
- **Fallback (b), a live view**: a virtual display plus VNC over SSH. Xvfb will not work, because Isaac Gym's Vulkan crashes under it. It needs a real headless Xorg on the NVIDIA driver, a VNC server on that display, `ssh -L 5901:localhost:5901`, and `run.py` started without `--headless`.
  ```bash
  # unverified: nvidia-xconfig --use-display-device=None --virtual=1920x1080, then Xorg :1 and x11vnc -display :1
  ```
