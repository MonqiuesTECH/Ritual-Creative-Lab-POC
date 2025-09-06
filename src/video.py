from pathlib import Path
import tempfile, io
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, VideoClip, ImageClip, CompositeVideoClip,
    TextClip
)

TARGET_W, TARGET_H, TARGET_FPS = 720, 1280, 8  # vertical

def write_temp_bytes(file, suffix=".bin") -> Path:
    data = file.read() if hasattr(file, "read") else file
    p = Path(tempfile.mkstemp(suffix=suffix)[1])
    with open(p, "wb") as f:
        f.write(data)
    return p

def _ease_in_out_cos(t: float) -> float:
    return 0.5 - 0.5*np.cos(np.pi*t)

def animate_image_to_video(image_path: Path, seconds=6, fps=8) -> Path:
    """Ken-Burns: subtle zoom + drift. CPU-only."""
    img = Image.open(image_path).convert("RGB")
    # working canvas ~ 720x960 then we’ll letterbox to 720x1280 later
    base_w, base_h = 720, 960

    def make_frame(t):
        tt = t / max(seconds, 1e-6)
        z = 1.05 + 0.12*_ease_in_out_cos(tt)      # 1.05→1.17
        y_off = int((tt-0.5) * 60)                # slight vertical drift
        resized = img.resize((int(base_w*z), int(base_h*z)), Image.LANCZOS)
        canvas = Image.new("RGB", (base_w, base_h), (12,12,14))
        canvas.paste(resized, ((base_w-resized.width)//2, (base_h-resized.height)//2 + y_off))
        # subtle vignette
        arr = np.asarray(canvas, dtype=np.float32)
        ys, xs = np.mgrid[0:base_h, 0:base_w]
        cx, cy = base_w/2, base_h/2
        rad  = np.sqrt((xs-cx)**2/(base_w*0.6)**2 + (ys-cy)**2/(base_h*0.85)**2)
        mask = np.clip(1.0 - rad, 0.0, 1.0)**2
        arr = (arr * (0.86 + 0.14*mask[...,None])).clip(0,255).astype(np.uint8)
        return arr

    clip = VideoClip(make_frame, duration=seconds).set_fps(fps)
    out = Path(tempfile.mkstemp(suffix=".mp4")[1])
    clip.write_videofile(str(out), codec="libx264", audio=False, fps=fps, preset="veryfast", threads=2)
    clip.close()
    return out

def letterbox_720x1280(src_path: Path, fps: Optional[int]=None) -> Path:
    """Ensure vertical 720x1280 with letterbox/pad, keep original AR."""
    clip = VideoFileClip(str(src_path))
    fps = fps or getattr(clip, "fps", TARGET_FPS) or TARGET_FPS
    clip_scaled = clip.resize(height=TARGET_H) if clip.h < TARGET_H else clip.resize(width=TARGET_W)
    # center on solid background
    bg = ImageClip(np.full((TARGET_H, TARGET_W, 3), (20,20,22), dtype=np.uint8), duration=clip.duration)
    comp = CompositeVideoClip([bg, clip_scaled.set_position(("center","center"))], size=(TARGET_W, TARGET_H))
    out = Path(tempfile.mkstemp(suffix=".mp4")[1])
    comp.write_videofile(str(out), codec="libx264", audio=False, fps=fps, preset="veryfast", threads=2)
    clip.close(); comp.close()
    return out

def _safe_text(txt: str) -> str:
    return (txt or "").strip()

def compose_vertical_ad(src_path: Path, title_text: str, subtitle_text: str, footer_text: str) -> Path:
    """Add headline/subtitle and a footer brand bar."""
    base = VideoFileClip(str(src_path))
    dur = base.duration

    # Title / subtitle using TextClip (fallback to PIL if ImageMagick unavailable)
    def _text_clip(t, size, color, stroke_color=None, fontsize=48, font="DejaVu-Sans"):
        try:
            return TextClip(_safe_text(t), fontsize=fontsize, color=color, font=font, method="caption",
                            align="center", size=size, stroke_color=stroke_color, stroke_width=1)
        except Exception:
            # simple fallback via PIL rendered onto an ImageClip
            img = Image.new("RGBA", size, (0,0,0,0))
            draw = ImageDraw.Draw(img)
            # crude font sizing
            draw.text((size[0]//2, size[1]//2), _safe_text(t), fill=color, anchor="mm")
            return ImageClip(np.array(img))

    title_clip = _text_clip(title_text, (680, None), "white", fontsize=54).set_position(("center", 80)).set_duration(dur)
    subtitle_clip = _text_clip(subtitle_text, (680, None), "#E0E0E0", fontsize=30).set_position(("center", 150)).set_duration(dur)

    # Footer brand bar
    bar_h = 56
    bar = ImageClip(np.full((bar_h, TARGET_W, 3), (10,10,12), dtype=np.uint8), duration=dur).set_position((0, TARGET_H-bar_h))
    footer = _text_clip(footer_text, (TARGET_W-40, bar_h), "#B0B0B0", fontsize=26).set_position(("center", TARGET_H-bar_h//2-2)).set_duration(dur)

    comp = CompositeVideoClip([base, title_clip, subtitle_clip, bar, footer], size=(TARGET_W, TARGET_H))
    out = Path("zari_ad.mp4").absolute()
    comp.write_videofile(str(out), codec="libx264", audio=False, fps=getattr(base, "fps", TARGET_FPS) or TARGET_FPS, preset="veryfast", threads=2)
    base.close(); comp.close()
    return out

def human_filesize(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit=="B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"
