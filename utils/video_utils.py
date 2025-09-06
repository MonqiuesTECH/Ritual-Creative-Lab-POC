from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
import tempfile, subprocess

import numpy as np
import imageio.v3 as iio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

def write_safe_mp4_from_frames(
    frames: List[np.ndarray] | List[Image.Image],
    out_path: str | Path,
    fps: int = 8,
    size: Tuple[int,int] = (720,1280),
    pad_hex: str = "0x141416",
) -> str:
    """H.264/yuv420p/baseline@3.0 + +faststart to avoid MoviePy/browser issues."""
    out_path = str(out_path)
    if len(frames) and isinstance(frames[0], Image.Image):
        frames = [np.array(f.convert("RGB")) for f in frames]  # type: ignore

    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    iio.imwrite(tmp, np.asarray(frames), fps=fps)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    w,h = size
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(tmp),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_hex}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-profile:v", "baseline", "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
        "-movflags", "+faststart",
        "-an", out_path
    ]
    subprocess.run(cmd, check=True)
    try: Path(tmp).unlink()
    except Exception: pass
    return out_path

def kenburns_from_image_frames(
    image: Image.Image,
    seconds: int = 4,
    fps: int = 8,
    portrait_box: Tuple[int,int] = (720,960),
    bg: Tuple[int,int,int] = (20,20,22),
) -> List[np.ndarray]:
    W,H = 720,1280
    inner_w, inner_h = portrait_box
    n = seconds*fps
    frames = []
    for i in range(n):
        t = i/max(n-1,1)
        import math
        ease = 0.5 - 0.5*math.cos(math.pi*t)
        z = 1.04 + 0.10*ease
        y_off = int((t-0.5)*40)
        base = image.resize((int(inner_w*z), int(inner_h*z)), Image.LANCZOS)
        portrait = Image.new("RGB",(inner_w, inner_h),(12,12,14))
        portrait.paste(base,((portrait.width-base.width)//2,(portrait.height-base.height)//2 + y_off))
        tall = Image.new("RGB",(W,H),bg)
        tall.paste(portrait, ((W-portrait.width)//2, (H-portrait.height)//2))
        frames.append(np.array(tall))
    return frames

def letterbox_clip_with_overlay(
    in_file,
    out_path: str | Path,
    text_top: Optional[str] = None,
    text_bottom: Optional[str] = None,
    size: Tuple[int,int] = (720,1280),
    fps: int = 8,
):
    W,H = size
    frames = []
    import tempfile
    from pathlib import Path
    # support Streamlit UploadedFile or path
    src_path = None
    if hasattr(in_file, "getvalue"):
        tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
        with open(tmp, "wb") as f: f.write(in_file.getvalue())
        src_path = str(tmp)
    else:
        src_path = str(in_file)

    for fr in iio.imiter(src_path):
        im = Image.fromarray(fr)
        s = min(W/im.width, H/im.height)
        nw, nh = max(1,int(im.width*s)), max(1,int(im.height*s))
        im = im.resize((nw,nh), Image.LANCZOS)
        canvas = Image.new("RGB",(W,H),(20,20,22))
        canvas.paste(im, ((W-nw)//2,(H-nh)//2))
        draw = ImageDraw.Draw(canvas)
        try:
            font_h = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
            font_b = ImageFont.truetype("DejaVuSans.ttf", 28)
        except Exception:
            font_h = ImageFont.load_default(); font_b = ImageFont.load_default()
        if text_top:
            draw.text((24,24), text_top, font=font_h, fill=(255,255,255))
        if text_bottom:
            fb_h = 54
            draw.rectangle([(0,H-fb_h),(W,H)], fill=(17,24,39))
            draw.text((24, H-fb_h+14), text_bottom, font=font_b, fill=(255,255,255))
        frames.append(np.array(canvas))
    write_safe_mp4_from_frames(frames, out_path, fps=fps, size=size)
