"""COLMAP PatchMatch depth maps -> a capture-style folder for tsdf_mesh.py.

    python colmap_depth.py DENSE CAPTURE OUT

DENSE is a COLMAP dense workspace (image_undistorter at the capture's own size, then patch_match_stereo).
Writes OUT/depth/NAME (uint16 mm from NAME.geometric.bin) and links CAPTURE's images, sparse and test.txt.
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def read_array(path):
    """COLMAP's .bin depth map: an ASCII 'w&h&c&' header, then float32 in column-major order."""
    with open(path, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = map(int, head.split(b"&")[:3])
        a = np.fromfile(f, np.float32)
    return a.reshape((w, h, c), order="F").transpose(1, 0, 2).squeeze()


def main():
    dense, cap, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    (out / "depth").mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((dense / "stereo/depth_maps").glob("*.geometric.bin")):
        z = np.nan_to_num(read_array(p)).clip(0, 60)
        Image.fromarray((z * 1000).round().astype(np.uint16)).save(out / "depth" / p.name[:-len(".geometric.bin")])
        n += 1
    for name in ("images", "sparse", "test.txt"):
        if not (out / name).exists():
            os.symlink(cap / name, out / name)
    print(f"{n} depth maps -> {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        import tempfile
        a = np.arange(12, dtype=np.float32).reshape(3, 4)  # h 3, w 4
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"4&3&1&")
            f.write(a.tobytes())  # (w, h) column-major = x fastest = the image row by row
        assert np.array_equal(read_array(f.name), a)
        print("ok")
    else:
        main()
