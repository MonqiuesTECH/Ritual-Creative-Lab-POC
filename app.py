import streamlit as st
from pathlib import Path
from src.ui import brand_form, persona_form, footer_brand_bar
from src.video import (
    write_temp_bytes, animate_image_to_video, compose_vertical_ad,
    letterbox_720x1280, human_filesize
)

st.set_page_config(page_title="Ritual Ads • Powered by ZARI", page_icon="✨", layout="wide")
st.title("Ritual Ads • Powered by ZARI")

with st.sidebar:
    st.header("Brand & Campaign")
    brand = brand_form()
    personas = persona_form()

    st.divider()
    st.caption("Footer brand bar")
    footer_txt = footer_brand_bar()

st.subheader("1) Base creative")
tab1, tab2 = st.tabs(["Upload base MP4", "Upload hero image → auto-animate"])
base_clip_path = None

with tab1:
    mp4 = st.file_uploader("Upload MP4 (720×1280 preferred, any OK)", type=["mp4"])
    if mp4:
        base_clip_path = write_temp_bytes(mp4, suffix=".mp4")
        st.video(str(base_clip_path))

with tab2:
    img = st.file_uploader("Upload hero image (JPG/PNG)", type=["jpg","jpeg","png"])
    seconds = st.slider("Seconds", 3, 12, 6, 1)
    fps = st.slider("FPS", 6, 24, 8, 2)
    if st.button("Make motion clip from image") and img:
        img_path = write_temp_bytes(img, suffix=".png")
        with st.spinner("Animating…"):
            base_clip_path = animate_image_to_video(img_path, seconds=seconds, fps=fps)
        st.success("Base motion clip created")
        st.video(str(base_clip_path))

st.subheader("2) Compose ZARI ad (720×1280)")
title = st.text_input("Hero headline", value=brand.get("primary_goal","Engagement"))
subtitle = st.text_input("Support line", value=brand.get("voice_tone","Bold • Modern • Human-centered"))

if st.button("Compose Final Ad") and base_clip_path:
    with st.spinner("Composing…"):
        # Letterbox (or downscale) and add brand chrome
        lb = letterbox_720x1280(base_clip_path)
        out_path = compose_vertical_ad(
            lb,
            title_text=title.strip(),
            subtitle_text=subtitle.strip(),
            footer_text=footer_txt.strip()
        )
    size_mb = human_filesize(Path(out_path).stat().st_size)
    st.success(f"Done → {out_path.name} ({size_mb})")
    st.video(str(out_path))
    st.download_button("Download MP4", out_path.read_bytes(), file_name=out_path.name, mime="video/mp4")
elif st.button("Compose Final Ad") and not base_clip_path:
    st.warning("Upload a base MP4 or make a motion clip from an image first.")
