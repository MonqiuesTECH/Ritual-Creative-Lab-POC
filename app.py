import io, json, zipfile, time
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from utils.video_utils import (
    kenburns_from_image_frames,
    write_safe_mp4_from_frames,
    letterbox_clip_with_overlay,
)
from utils.text_utils import (
    generate_segments,
    variant_copy_for_segment,
)
from utils.design_utils import (
    parse_hex, gradient_bg, wrap_text, ensure_dir
)

APP_TITLE = "Ritual Creative Lab — POC (Powered by ZARI)"

# ---------- sidebar: brand & campaign ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🪄", layout="wide")
st.sidebar.header("Brand & Campaign")

if st.sidebar.button("Load Demo Campaign"):
    st.session_state["brand_desc"] = "ZARI builds AI operators for founders and lean teams — turning repeatable work into reliable rituals that run themselves, stay on-brand, and deliver measurable ROI."
    st.session_state["goal"] = "Engagement"
    st.session_state["voice"] = "Bold, modern, human-centered"
    st.session_state["personas"] = "Seed-stage SaaS founders (2–20 employees)\nShopify/DTC operators (>$50k/mo)\nService business owners who book appointments"

brand_desc = st.sidebar.text_area(
    "Brand description (who/why/ritual):",
    value=st.session_state.get("brand_desc", ""),
    height=100,
)

goal = st.sidebar.selectbox("Primary goal", ["Awareness", "Engagement", "Signups/Leads", "Sales/Conversions"],
                            index=["Awareness","Engagement","Signups/Leads","Sales/Conversions"].index(
                                st.session_state.get("goal","Engagement"))
                            )

voice = st.sidebar.text_input("Voice / Tone", value=st.session_state.get("voice", "Bold, modern, human-centered"))

personas_txt = st.sidebar.text_area(
    "Personas (one per line)",
    value=st.session_state.get("personas", "Seed-stage SaaS founders (2–20 employees)\nShopify/DTC operators (>$50k/mo)"),
    height=100,
)

seg_count = st.sidebar.slider("Number of persona segments", 1, 6, 2)
variants_per = st.sidebar.slider("Variants per segment", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Brand Kit (optional)")
primary_hex = st.sidebar.color_picker("Primary", "#7C3AED")
secondary_hex = st.sidebar.color_picker("Secondary", "#111827")
footer_text = st.sidebar.text_input("Footer brand bar", "Ritual Ads • Powered by ZARI")
poster_template = st.sidebar.selectbox("Poster template", ["Hero", "Minimal"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Video (optional)")
enable_video = st.sidebar.toggle("Generate videos (MP4)", True)
video_mode = st.sidebar.selectbox("Source", ["Auto (Ken-Burns from poster)", "Upload base AI clip"], index=0)
video_fps = st.sidebar.selectbox("FPS", [6, 8, 12], index=1)
video_seconds = st.sidebar.slider("Duration (seconds)", 3, 8, 4)
upload_file = None
if enable_video and video_mode == "Upload base AI clip":
    upload_file = st.sidebar.file_uploader("Upload MP4 clip", type=["mp4"])

st.sidebar.markdown("---")
st.sidebar.caption("Free • Local ops • No API keys")

# ---------- main ----------
st.title(APP_TITLE)
tabs = st.tabs(["Plan", "Generate", "Review & Download"])

# ---------- PLAN ----------
with tabs[0]:
    st.subheader("Creative Plan")
    st.markdown(
        """
- Cluster personas ➜ generate copy per segment ➜ auto-create poster visuals.
- Everything runs locally with open-source libs on CPU.
- Outputs: headline, body, CTA, suggested channel, and PNG posters per variant.
- (Optional) short MP4 per variant, encoded for social (H.264/yuv420p/+faststart).
        """
    )
    st.info("Tip: keep personas concrete (e.g., “CTOs at seed-stage SaaS”).")
    st.code(brand_desc or "Describe your brand here…", language="markdown")

# ---------- GENERATE ----------
with tabs[1]:
    st.subheader("Generate Ad Packages")
    run = st.button("Run Generator", use_container_width=True, type="primary")
    if run:
        t0 = time.time()
        # 1) derive segments
        raw_personas = [p.strip() for p in personas_txt.splitlines() if p.strip()]
        segments = generate_segments(raw_personas, seg_count)

        # prepare output structure
        artifacts_dir = Path("artifacts")
        posters_dir = artifacts_dir / "posters"
        videos_dir = artifacts_dir / "videos"
        ensure_dir(posters_dir); ensure_dir(videos_dir)

        primary = parse_hex(primary_hex); secondary = parse_hex(secondary_hex)

        packages: Dict[str, List[Dict]] = {}
        for i, seg in enumerate(segments, start=1):
            seg_key = f"segment_{i}"
            packages[seg_key] = []
            for v in range(1, variants_per + 1):
                # 2) copy
                copy = variant_copy_for_segment(
                    brand_desc=brand_desc,
                    goal=goal,
                    voice=voice,
                    persona=seg,
                    variant=v,
                )

                # 3) poster (PNG)
                W, H = 1080, 1350  # 4:5 portrait
                img = gradient_bg(W, H, primary, secondary)
                draw = ImageDraw.Draw(img)
                try:
                    font_h = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
                    font_b = ImageFont.truetype("DejaVuSans.ttf", 36)
                except Exception:
                    font_h = ImageFont.load_default()
                    font_b = ImageFont.load_default()

                pad = 64
                y = pad
                title = f"Unlock {seg}"
                y = wrap_text(draw, title, (pad, y), W - 2*pad, font_h, fill=(255,255,255))
                y += 12
                y = wrap_text(draw, copy["body"], (pad, y), W - 2*pad, font_b, fill=(232, 232, 232))
                y += 24
                y = wrap_text(draw, f"CTA: {copy['cta']}", (pad, y), W - 2*pad, font_b, fill=(255,255,255))

                # footer bar
                fb_h = 56
                draw.rectangle([(0, H - fb_h), (W, H)], fill=secondary)
                draw.text((pad, H - fb_h + 14), footer_text, font=font_b, fill=(255,255,255))

                poster_path = posters_dir / f"{seg_key}_v{v}.png"
                img.save(poster_path)

                # 4) video (MP4)
                mp4_path = None
                if enable_video:
                    if video_mode == "Auto (Ken-Burns from poster)":
                        frames = kenburns_from_image_frames(
                            image=img,
                            seconds=video_seconds,
                            fps=video_fps,
                            portrait_box=(720, 960),  # inner content area
                            bg=(20,20,22)
                        )
                        mp4_path = videos_dir / f"{seg_key}_v{v}.mp4"
                        write_safe_mp4_from_frames(frames, mp4_path, fps=video_fps, size=(720,1280))
                    else:
                        if upload_file is not None:
                            mp4_path = videos_dir / f"{seg_key}_v{v}.mp4"
                            # overlay simple title + footer over uploaded clip
                            letterbox_clip_with_overlay(
                                in_file=upload_file,
                                out_path=mp4_path,
                                text_top=f"Unlock {seg}",
                                text_bottom=footer_text,
                                size=(720,1280),
                                fps=video_fps
                            )
                        else:
                            st.warning("Upload a base clip or switch source to Auto (Ken-Burns).")

                packages[seg_key].append({
                    "persona": seg,
                    "variant": v,
                    "copy": copy,
                    "poster": str(poster_path),
                    "video": str(mp4_path) if mp4_path else None,
                })

        st.success(f"Generated {sum(len(v) for v in packages.values())} packages in {time.time()-t0:.1f}s.")
        st.session_state["packages"] = packages

        # quick preview grid
        cols = st.columns(variants_per)
        i = 0
        for seg_key, items in packages.items():
            for item in items:
                with cols[i % variants_per]:
                    st.image(item["poster"], caption=f"{seg_key} • V{item['variant']}")
                    if item["video"]:
                        with open(item["video"], "rb") as f:
                            st.video(f.read())
                i += 1

# ---------- REVIEW & DOWNLOAD ----------
with tabs[2]:
    st.subheader("Review & Download")
    pkgs = st.session_state.get("packages")
    if not pkgs:
        st.info("Run the generator first.")
    else:
        # show copy
        for seg_key, items in pkgs.items():
            st.markdown(f"### {seg_key.replace('_',' ').title()}")
            for it in items:
                st.markdown(f"**Variant {it['variant']}** — *{it['persona']}*")
                st.write(f"**Headline:** {it['copy']['headline']}")
                st.write(f"**Body:** {it['copy']['body']}")
                st.write(f"**CTA:** {it['copy']['cta']}")
                st.caption(f"Poster: {Path(it['poster']).name}" + (f" • Video: {Path(it['video']).name}" if it['video'] else ""))

        # build ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            copy_json = {}
            for seg_key, items in pkgs.items():
                for it in items:
                    z.write(it["poster"], f"posters/{Path(it['poster']).name}")
                    if it["video"]:
                        z.write(it["video"], f"videos/{Path(it['video']).name}")
            # save copy JSON
            for seg_key, items in pkgs.items():
                copy_json[seg_key] = [{"variant": it["variant"], **it["copy"]} for it in items]
            z.writestr("copy.json", json.dumps(copy_json, indent=2))
        buf.seek(0)
        st.download_button(
            "Download ZIP (PNG + MP4 + copy.json)",
            data=buf.read(),
            file_name="ritual_creative_lab_poc.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
