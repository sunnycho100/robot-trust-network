"""ROS1 or ROS2 bag with a RealSense-style RGB-D camera -> a frames folder for reconstruct.py.

    python extract_bag.py BAG OUT_DIR [--stride 2] [--width 640]

Reads color, depth aligned to color, and camera_info (intrinsics). Writes OUT_DIR/color/*.jpg,
OUT_DIR/depth/*.png (uint16, unchanged units), OUT_DIR/intrinsics.json and timestamps.txt. The depth unit is
checked against the scene: --depth-scale is meters per raw unit (RealSense D4xx 0.001,
L515 often 0.00025).
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def image(msg):
    channels = {"rgb8": 3, "bgr8": 3, "16UC1": 1, "mono16": 1}[msg.encoding]
    dtype = np.uint16 if msg.encoding in ("16UC1", "mono16") else np.uint8
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    if msg.encoding == "rgb8":
        array = array[..., ::-1]  # OpenCV writes BGR
    return array.squeeze()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--color", default="/camera/color/image_raw")
    p.add_argument("--depth", default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--info", default="/camera/color/camera_info")
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    p.add_argument("--width", type=int, default=640, help="resize frames to this width (intrinsics follow)")
    p.add_argument("--depth-scale", type=float, default=0.001, help="meters per raw depth unit")
    a = p.parse_args()
    (a.out / "color").mkdir(parents=True, exist_ok=True)
    (a.out / "depth").mkdir(parents=True, exist_ok=True)

    with AnyReader([a.bag]) as reader:
        wanted = [c for c in reader.connections if c.topic in (a.color, a.depth, a.info)]
        colors, depths, info = {}, {}, None
        for connection, stamp, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, connection.msgtype)
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if connection.topic == a.info:
                info = info or msg
            else:  # shrink on read, so a long bag does not fill memory
                frame = image(msg)
                size = (a.width, round(msg.height * a.width / msg.width))
                resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if connection.topic == a.color else cv2.INTER_NEAREST)
                (colors if connection.topic == a.color else depths)[t] = resized
    if info is None or not colors or not depths:
        raise SystemExit("bag is missing color, depth or camera_info on the given topics")

    depth_times = np.array(sorted(depths))
    scale = a.width / info.width
    kept, stamps = 0, []
    for i, t in enumerate(sorted(colors)):
        if i % a.stride:
            continue
        nearest = depth_times[np.abs(depth_times - t).argmin()]
        if abs(nearest - t) > 20_000_000:  # more than 20 ms apart: not the same moment
            continue
        cv2.imwrite(str(a.out / "color" / f"{kept:06d}.jpg"), colors[t])
        cv2.imwrite(str(a.out / "depth" / f"{kept:06d}.png"), depths[nearest])
        stamps.append(t)
        kept += 1
    np.savetxt(a.out / "timestamps.txt", np.array(stamps, dtype=np.int64), fmt="%d")  # ns, for lidar_poses.py
    K = np.array(getattr(info, "k", None) if hasattr(info, "k") else info.K).reshape(3, 3)  # ROS2 k, ROS1 K
    intrinsics = {"width": a.width, "height": round(info.height * scale), "fx": K[0, 0] * scale, "fy": K[1, 1] * scale,
                  "cx": K[0, 2] * scale, "cy": K[1, 2] * scale, "depth_scale": a.depth_scale}
    (a.out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    sample = np.concatenate([d[d > 0] for d in list(depths.values())[::50]])
    print(f"{kept} frames, {intrinsics['width']}x{intrinsics['height']}, fx={intrinsics['fx']:.1f} -> {a.out}")
    print(f"depth median {np.median(sample) * a.depth_scale:.2f} m, 95th percentile {np.percentile(sample, 95) * a.depth_scale:.2f} m "
          f"(with --depth-scale {a.depth_scale}); a room should read about 1 to 8 m")


if __name__ == "__main__":
    main()
