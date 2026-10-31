"""SfM init on known poses: SIFT, sequential matching, COLMAP triangulation -> OUT/sparse/0/points3D.txt.

    python tri_init.py CAPTURE OUT      (pycolmap 4.2 venv; CPU is fine for 240 frames at 1280x720)

The fair stand-in for a real capture's SfM points (the critic's headline init): the capture's own points3D.txt is
GT depth and would leak geometry. Poses and intrinsics stay exact; only the points come from image matching.
Held-out frames (CAPTURE/test.txt) are left out of matching so they never seed the splat. OUT gets the capture's
cameras.txt and images.txt (all frames, so 3DGRUT keeps its test split) plus the triangulated points.
"""
import shutil
import sys
from pathlib import Path
import pycolmap


def read_images(path):
    """images.txt -> {name: (qvec, tvec)}; tolerant of the empty POINTS2D lines in the known-pose model."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 10 and not line.startswith("#"):  # ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            out[p[9]] = (p[1:5], p[5:8])
    return out


def main():
    cap, out = Path(sys.argv[1]), Path(sys.argv[2])
    sp, work = out / "sparse/0", out / "tri"
    shutil.rmtree(out, ignore_errors=True)
    sp.mkdir(parents=True), work.mkdir()
    test = {l.strip() for l in open(cap / "test.txt") if l.strip()}
    poses = read_images(cap / "sparse/0/images.txt")
    train = sorted(n for n in poses if n not in test)
    cam = next(l for l in open(cap / "sparse/0/cameras.txt") if l.strip() and not l.startswith("#")).split()
    assert cam[1] == "PINHOLE", cam

    db = work / "db.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model, reader.camera_params = "PINHOLE", ",".join(cam[4:8])
    pycolmap.extract_features(db, cap / "images", image_names=train, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 20
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)

    # known-pose model keyed by the database's own ids (point_triangulator matches cameras and images by id)
    d = pycolmap.Database.open(db)
    ids = {im.name: (im.image_id, im.camera_id) for im in d.read_all_images()}
    d.close()
    assert len(ids) == len(train) and len({c for _, c in ids.values()}) == 1, (len(ids), len(train))
    m = work / "known"
    m.mkdir()
    cid = next(iter(ids.values()))[1]
    (m / "cameras.txt").write_text(f"{cid} " + " ".join(cam[1:]) + "\n")
    with open(m / "images.txt", "w") as f:
        for n in train:
            q, t = poses[n]
            f.write(f"{ids[n][0]} {' '.join(q)} {' '.join(t)} {cid} {n}\n\n")
    (m / "points3D.txt").write_text("")
    rec = pycolmap.triangulate_points(pycolmap.Reconstruction(m), db, cap / "images", work / "out")
    print(f"[tri] {len(train)} train frames, {rec.num_points3D()} points, "
          f"mean track {rec.compute_mean_track_length():.2f}, mean reproj {rec.compute_mean_reprojection_error():.3f} px")
    with open(sp / "points3D.txt", "w") as f:
        for pid, p in rec.points3D.items():
            x, y, z = p.xyz
            r, g, b = p.color
            f.write(f"{pid} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} {p.error:.4f}\n")
    for n in ("cameras.txt", "images.txt"):
        shutil.copy(cap / "sparse/0" / n, sp / n)
    (out / "images").symlink_to((cap / "images").resolve())


if __name__ == "__main__":
    main()
