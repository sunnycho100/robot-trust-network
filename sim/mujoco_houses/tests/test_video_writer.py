import shutil
import sys

import numpy as np
import pytest

from unitree_demo.coop_lift import RawVideoWriter


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_records_without_tqdm(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tqdm", None)  # import tqdm now raises ImportError
    path = tmp_path / "clip.mp4"
    writer = RawVideoWriter(path, 64, 48, 10, total_frames=2)
    for _ in range(2):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.close()
    assert path.stat().st_size > 0
