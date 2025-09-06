import io, os, math, tempfile, time
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio.v3 as iio
import streamlit as st

st.set_page_config(page_title="Ritual Ads • ZARI Demo", page_icon="🎬", layout="wide")

# ---------- Small helpers ----------
def parse_lines(s: str) -> List[str]:
    return [ln.strip() for ln in s.splitlines() if ln.strip()]

def vignette(arr: np.ndarray, strength: float = 0.14) -> np.ndarray:
    """Subtle radial vignette for premium look."""
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    rad = np.sqrt(((xs - cx) ** 2) / (w * 0.6) ** 2 + ((ys - cy) ** 2) / (h * 0.85) ** 2)
    mask = np.clip(1.0 - rad, 0.0, 1.0) ** 2
    out = (arr.astype(np.float32) * (0.86 + strength * mask[..., None])).clip(0, 255).astype(np.uint8)
    return out

def ken_burns_frames(img: Image.Image, seconds: int = 6, fps: int = 8,
                     canvas_size: Tuple[int, int] = (720, 1280)) -> List[np.ndarray]:
    """Animate a single image with smooth zoom + slight vertical drift on a 720x1280 canvas."""
    W, H = canvas_size
    n = seconds * fps
    frames = []
    # base working area (taller box inside 720x1280)
    work_w, work_h = 720, 960

    def ease(t: float) -> float:
        return 0.5 - 0.5 * math.cos(math.pi * t)  # smooth in/out

    # Ensure the source covers our work area (avoid letterboxing inside the work box)
    # Resize while preserving aspect
    s = max(work_w / img.width, work_h / img.height)
    base_big = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                          Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)

    for i in range(n):
        t = i / max(n - 1, 1)
        z = 1.05 + 0.12 * ease(t)          # 1.05 → 1.17 zoom
        y_off = int((t - 0.5) * 60)        # small vertical drift

        # Zoom by resizing around center
        zw, zh = int(base_big.width * z), int(base_big.height * z)
        zoomed = base_big.resize((zw, zh),
                                 Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)

        # Paste into a 720x960 work canvas
        work = Image.new("RGB", (work_w, work_h), (12, 12, 14))
        work.paste(zoomed, ((work_w - zw) // 2, (work_h - zh) // 2 + y_off))

        # Lift to a 720x1280 tall canvas and center
        tall = Image.new("RGB", (W, H), (20, 20, 22))
        tall.paste(work, (0, (H - work_h) // 2))

        arr = np.array(tall)
        arr = vignette(arr, strength=0.14)
        frames.append(arr)
    return frames

def write_mp4(frames: List[np.ndarray], fps: int, path: str) -> None:
    iio.imwrite(path, np.array(frames), fps=fps)

def make_footer_bar(arr: np.ndarray, text: str, color_hex: str = "#111111") -> np.ndarray:
    """Add a branded footer bar with text."""
    h, w = arr.shape[:2]
    bar_h = int(h * 0.08)
    out = arr.copy()
    # bar
    bar = np.ones((bar_h, w, 3), dtype=np.uint8) * 255
    # tint bar with color
    c = tuple(int(color_hex.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    bar = (bar * 0.0 + np.array(c, dtype=np.uint8)).astype(np.uint8)
    out[h - bar_h : h, :, :] = bar

    # draw text
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        # Streamlit Cloud may not have fonts; default to PIL's built-in
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textlength(text, font=font), 12
    draw.text(((w - tw) / 2, h - bar_h + (bar_h - th) / 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)

# ---------- UI ----------
st.title("Ritual Ads • Powered by ZARI")
st.caption("Lite Templates • Simple demo — generate a vertical video from an image or upload a finished MP4.")

with st.form("zari_form"):
    st.subheader("Brand & Campaign")
    brand_desc = st.text_area("Brand description (who/why/ritual)", height=90,
                              placeholder="Who are we for? Why do we matter? What ritual are we building?")
    primary_goal = st.selectbox("Primary goal", ["Engagement", "Awareness", "Conversion"], index=0)
    voice_tone = st.text_input("Voice / Tone", value="Bold, modern, human-centered")
    personas_raw = st.text_area("Personas (one per line)", height=80,
                                placeholder="e.g., Fitness-curious millennials\nCTOs at seed-stage SaaS")
    num_segments = st.slider("Number of persona segments", 1, 6, 1)
    variants_per = st.slider("Variants per segment", 1, 3, 1)

    st.markdown("---")
    st.subheader("Brand Kit (optional)")
    colA, colB = st.columns(2)
    with colA:
        color_primary = st.color_picker("Primary", "#0ea5e9")
    with colB:
        color_secondary = st.color_picker("Secondary", "#22c55e")
    footer_on = st.checkbox("Footer brand bar", value=True)
    footer_text = "Ritual Ads • Powered by ZARI"

    st.markdown("---")
    st.subheader("Poster template")
    st.write("**Hero** + **Video (optional)**")
    resolution = st.selectbox("Resolution", ["720x1280 (fast)"], index=0)

    st.markdown("### Source for Video")
    src_choice = st.radio("Choose input", ["Upload image (AI animates it)", "Upload finished MP4"], index=0)

    up_img = None
    up_mp4 = None
    if src_choice == "Upload image (AI animates it)":
        up_img = st.file_uploader("Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    else:
        up_mp4 = st.file_uploader("MP4 video", type=["mp4"])

    submitted = st.form_submit_button("Generate / Preview")

if submitted:
    t0 = time.time()
    if src_choice == "Upload image (AI animates it)":
        if not up_img:
            st.error("Please upload a JPG/PNG image.")
            st.stop()
        img = Image.open(up_img).convert("RGB")
        frames = ken_burns_frames(img, seconds=6, fps=8, canvas_size=(720, 1280))
        if footer_on:
            frames = [make_footer_bar(fr, footer_text, color_primary) for fr in frames]

        with tempfile.TemporaryDirectory() as td:
            mp4_path = os.path.join(td, "zari_demo.mp4")
            write_mp4(frames, fps=8, path=mp4_path)
            st.success(f"Video generated in {time.time() - t0:.1f}s")
            st.video(mp4_path)
            st.download_button("Download MP4", data=open(mp4_path, "rb").read(),
                               file_name="zari_demo.mp4", mime="video/mp4")

    else:
        if not up_mp4:
            st.error("Please upload an MP4.")
            st.stop()
        # Just preview + let user download; optionally we could re-encode for size
        data = up_mp4.read()
        st.video(io.BytesIO(data))
        st.download_button("Download MP4", data=data, file_name="zari_demo.mp4", mime="video/mp4")

# ---------- “Copy mode” block (optional)—shows the ad copy you can paste elsewhere ----------
with st.expander("Copy mode • generated talking points"):
    personas = parse_lines(personas_raw)
    st.markdown(f"**Goal:** {primary_goal}")
    st.markdown(f"**Voice/Tone:** {voice_tone}")
    st.markdown("**Personas:**")
    if personas:
        for p in personas[:num_segments]:
            st.markdown(f"- {p}")
    else:
        st.markdown("- (none provided)")
    st.markdown(f"**Variants per segment:** {variants_per}")
    st.caption("Tip: Use this section as the copy source when you export your assets.")
