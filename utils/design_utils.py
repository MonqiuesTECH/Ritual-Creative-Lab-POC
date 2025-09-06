from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def ensure_dir(p: Path | str):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_hex(h: str) -> Tuple[int,int,int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def gradient_bg(w: int, h: int, c1: tuple[int,int,int], c2: tuple[int,int,int]) -> Image.Image:
    top = np.zeros((h,w,3), dtype=np.float32)
    for y in range(h):
        t = y/max(h-1,1)
        top[y,:,0] = c1[0]*(1-t)+c2[0]*t
        top[y,:,1] = c1[1]*(1-t)+c2[1]*t
        top[y,:,2] = c1[2]*(1-t)+c2[2]*t
    noise = (np.random.rand(h,w,1)*8-4)
    top = np.clip(top+noise,0,255).astype(np.uint8)
    return Image.fromarray(top,"RGB")

def wrap_text(draw: ImageDraw.ImageDraw, text: str, xy, max_w: int, font: ImageFont.ImageFont, fill=(255,255,255)):
    x,y = xy
    words = text.split()
    line = ""
    for w in words:
        test = (line+" "+w).strip()
        if draw.textlength(test, font=font) <= max_w:
            line = test
            continue
        draw.text((x,y), line, font=font, fill=fill)
        y += font.size + 6
        line = w
    if line:
        draw.text((x,y), line, font=font, fill=fill)
        y += font.size + 6
    return y
