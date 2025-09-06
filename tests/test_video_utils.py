import numpy as np
from utils.video_utils import write_safe_mp4_from_frames

def test_write_safe_mp4(tmp_path):
    frames = [np.zeros((128,72,3), dtype=np.uint8) for _ in range(8)]
    out = tmp_path / "t.mp4"
    p = write_safe_mp4_from_frames(frames, out, fps=8, size=(72,128))
    assert out.exists()
