import io, json, zipfile, time, os
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.auth import require_password
from utils.storage import get_storage
from utils.design_utils import parse_hex, gradient_bg, wrap_text, ensure_dir
from utils.text_utils import (
    cluster_personas,
    generate_variants_for_segment,
    score_copy_package,
)
from utils.video_utils import (
    kenburns_from_image_frames,
    write_safe_mp4_from_frames,
    letterbox_clip_with_overlay,
)

APP_TITLE = "Ritual Creative Lab — POC (Powered by ZARI)"
ARTIFACTS = Path("artifacts"); ensure_dir(ARTIFACTS)

# ---------- optional password gate ----------
require_password(st.secrets.get("APP_PASSWORD"))

# ---------- page ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🪄", layout="wide")
st.sidebar.header("Brand & Campaign")

# demo loader
if st.sidebar.button("Load Demo Campaign"):
    st.session_state["brand_desc"] = (
        "ZARI builds AI operators for founders and lean teams — "
        "turning repeatable work into reliable rituals that run themselves, "
        "stay on-brand, and deliver measurable ROI."
    )
    st.session_state["goal"] = "Signups/Leads"
    st.session_state["voice"] = "Bold, modern, human-centered"
    st.session_state["personas"] = (
        "Seed-stage SaaS founders (2–20 employees)\n"
        "Shopify/DTC operators (>$50k/mo)\n"
        "Service business owners who book appointments\n"
        "Creators launching digital courses\n"
        "Eco-conscious urban professionals"
    )

brand_desc = st.sidebar.text_area(
    "Brand description (who/why/ritual):",
    value=st.session_state.get("brand_desc", ""),
    height=100,
)
goal = st.sidebar.selectbox(
    "Primary goal",
    ["Awareness","Engagement","Signups/Leads","Sales/Conversions"],
    index=["Awareness","Engagement","Signups/Leads","Sales/Conversions"].index(
        st.session_state.get("goal","Signups/Leads"))
)
voice = st.sidebar.text_input("Voice / Tone", value=st.session_state.get("voice", "Bold, modern, human-centered"))
personas_txt = st.sidebar.text_area(
    "Personas (one per line)",
    value=st.session_state.get("personas", ""),
    height=120,
)
seg_count = st.sidebar.slider("Number of persona segments", 1, 6, 3)
variants_per = st.sidebar.slider("Variants per segment", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Brand Kit (optional)")
primary_hex = st.sidebar.color_picker("Primary", "#7C3AED")
secondary_hex = st.sidebar.color_picker("Secondary", "#111827")
footer_text = st.sidebar.text_input("Footer brand bar", "Ritual Ads • Powered by ZARI")
poster_template = st.sidebar.selectbox("Poster template", ["Hero","Minimal"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Video (optional)")
enable_video = st.sidebar.toggle("Generate videos (MP4)", True)
video_mode = st.sidebar.selectbox("Source", ["Auto (Ken-Burns from poster)","Upload base AI clip"], index=0)
video_fps = st.sidebar.selectbox("FPS", [6,8,12], index=1)
video_seconds = st.sidebar.slider("Duration (seconds)", 3, 8, 4)
upload_file = None
if enable_video and video_mode == "Upload base AI clip":
    upload_file = st.sidebar.file_uploader("Upload MP4 clip", type=["mp4"])

st.sidebar.markdown("---")
st.sidebar.caption("Free • Local models optional • CPU only")

# ---------- main ----------
st.title(APP_TITLE)
tabs = st.tabs(["Plan","Generate","Review & Download","Metrics"])

# ---------- PLAN ----------
with tabs[0]:
    st.subheader("Creative Plan")
    st.markdown(
        """
- Cluster personas ➜ derive audience **segments** (TF-IDF + KMeans).
- For each segment, create **multiple copy variants**.
  - If available: **FLAN-T5-small** on CPU for LLM-quality copy.
  - Else: deterministic templates.
- Evaluate: **readability**, **brand keyword hit**, **length adherence**, **guardrails**.
- Output: posters (PNG) + short MP4s, plus **report.json** with scores & “best variant”.
        """
    )
    st.code(brand_desc or "Describe your brand here…", language="markdown")

# ---------- GENERATE ----------
with tabs[1]:
    st.subheader("Generate Ad Packages")
    run = st.button("Run Generator", use_container_width=True, type="primary")
    if run:
        t0 = time.time()
        personas = [p.strip() for p in personas_txt.splitlines() if p.strip()]
        if not personas:
            st.warning("Add at least one persona."); st.stop()

        # 1) cluster personas -> segments
        segments = cluster_personas(personas, seg_count)

        # 2) storage driver
        storage = get_storage()  # LocalDisk by default

        primary = parse_hex(primary_hex); secondary = parse_hex(secondary_hex)

        packages: Dict[str, List[Dict]] = {}
        poster_dir = ARTIFACTS / "posters"; poster_dir.mkdir(parents=True, exist_ok=True)
        video_dir  = ARTIFACTS / "videos";  video_dir.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0.0)
        step = 0
        total = len(segments) * variants_per

        for i, seg in enumerate(segments, start=1):
            seg_key = f"segment_{i}"
            packages[seg_key] = []
            # 3) generate N variants (LLM if available, else template) + score each
            variants = generate_variants_for_segment(
                brand_desc=brand_desc, goal=goal, voice=voice, persona_seg=seg, n_variants=variants_per
            )

            for v_idx, copy in enumerate(variants, start=1):
                # 4) poster (PNG)
                W, H = 1080, 1350
                img = gradient_bg(W, H, primary, secondary)
                draw = ImageDraw.Draw(img)
                try:
                    font_h = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
                    font_b = ImageFont.truetype("DejaVuSans.ttf", 36)
                except Exception:
                    font_h = ImageFont.load_default(); font_b = ImageFont.load_default()

                pad = 64
                y = pad
                title = copy["headline"] or f"Unlock {seg}"
                y = wrap_text(draw, title, (pad, y), W-2*pad, font_h, (255,255,255))
                y += 12
                y = wrap_text(draw, copy["body"], (pad, y), W-2*pad, font_b, (232,232,232))
                y += 24
                y = wrap_text(draw, f"CTA: {copy['cta']}", (pad, y), W-2*pad, font_b, (255,255,255))
                fb_h = 56
                draw.rectangle([(0,H-fb_h),(W,H)], fill=secondary)
                draw.text((pad, H-fb_h+14), footer_text, font=font_b, fill=(255,255,255))

                poster_path = poster_dir / f"{seg_key}_v{v_idx}.png"
                img.save(poster_path)

                # 5) optional video
                mp4_path = None
                if enable_video:
                    if video_mode == "Auto (Ken-Burns from poster)":
                        frames = kenburns_from_image_frames(img, seconds=video_seconds, fps=video_fps, portrait_box=(720,960))
                        mp4_path = video_dir / f"{seg_key}_v{v_idx}.mp4"
                        write_safe_mp4_from_frames(frames, mp4_path, fps=video_fps, size=(720,1280))
                    else:
                        if upload_file is not None:
                            mp4_path = video_dir / f"{seg_key}_v{v_idx}.mp4"
                            letterbox_clip_with_overlay(
                                in_file=upload_file, out_path=mp4_path, text_top=title, text_bottom=footer_text,
                                size=(720,1280), fps=video_fps
                            )

                # 6) scoring & guardrails
                score = score_copy_package(copy, brand_desc, goal)

                packages[seg_key].append({
                    "persona_segment": seg,
                    "variant": v_idx,
                    "copy": copy,
                    "poster": str(poster_path),
                    "video": str(mp4_path) if mp4_path else None,
                    "score": score
                })

                step += 1
                progress.progress(step/total)

        st.session_state["packages"] = packages
        st.success(f"Generated {sum(len(v) for v in packages.values())} packages in {time.time()-t0:.1f}s.")

        # preview grid with best variant badges
        cols = st.columns(variants_per)
        for seg_key, items in packages.items():
            # choose best by score.total
            best = max(items, key=lambda it: it["score"]["total"])
            st.markdown(f"### {seg_key.replace('_',' ').title()} — **Best: V{best['variant']} ({best['score']['total']})**")
            for it in items:
                c = cols[(it["variant"]-1) % variants_per]
                with c:
                    st.image(it["poster"], caption=f"V{it['variant']} | score {it['score']['total']}")
                    if it["video"]:
                        with open(it["video"], "rb") as f: st.video(f.read())

# ---------- REVIEW & DOWNLOAD ----------
with tabs[2]:
    st.subheader("Review & Download")
    pkgs: Optional[Dict[str, List[Dict]]] = st.session_state.get("packages")
    if not pkgs:
        st.info("Run the generator first.")
    else:
        # show details
        for seg_key, items in pkgs.items():
            st.markdown(f"### {seg_key.replace('_',' ').title()}")
            for it in items:
                st.markdown(f"**Variant {it['variant']}** — *{it['persona_segment']}*")
                st.write(f"**Headline:** {it['copy']['headline']}")
                st.write(f"**Body:** {it['copy']['body']}")
                st.write(f"**CTA:** {it['copy']['cta']}")
                st.json(it["score"])

        # produce ZIP & report.json (scores + best)
        report = {}
        for seg_key, items in pkgs.items():
            best = max(items, key=lambda it: it["score"]["total"])
            report[seg_key] = {
                "best_variant": best["variant"],
                "scores": [{ "variant": it["variant"], **it["score"] } for it in items]
            }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for seg_key, items in pkgs.items():
                for it in items:
                    z.write(it["poster"], f"posters/{Path(it['poster']).name}")
                    if it["video"]:
                        z.write(it["video"], f"videos/{Path(it['video']).name}")
            z.writestr("copy.json", json.dumps({
                seg: [{ "variant": it["variant"], **it["copy"] } for it in items ]
                for seg, items in pkgs.items()
            }, indent=2))
            z.writestr("report.json", json.dumps(report, indent=2))
        buf.seek(0)
        st.download_button(
            "Download ZIP (PNGs + MP4s + copy.json + report.json)",
            data=buf.read(),
            file_name="ritual_creative_lab_poc.zip",
            mime="application/zip",
            use_container_width=True, type="primary",
        )

# ---------- METRICS ----------
with tabs[3]:
    st.subheader("Metrics (session)")
    pkgs = st.session_state.get("packages", {})
    total_variants = sum(len(v) for v in pkgs.values()) if pkgs else 0
    st.metric("Segments", len(pkgs))
    st.metric("Variants", total_variants)
    st.caption("Simple in-app metrics. Hook to S3/OTel easily via `utils/storage.py` if creds exist.")
