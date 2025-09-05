# Ritual Creative Lab — POC (Powered by ZARI)
# Clean build: posters + optional MP4 video (auto slideshow or branded upload)
# Free, CPU-only. No API keys.

import hashlib
import io
import json
import os
import random
import re
import tempfile
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

APP_TITLE = "Ritual Creative Lab — POC (Powered by ZARI)"
BRAND_BAR_DEFAULT = "Ritual Ads • Powered by ZARI"
RANDOM_SEED = 42

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
random.seed(RANDOM_SEED)

# ---------------- Optional models (kept lightweight) ---------------- #
@st.cache_resource(show_spinner=False)
def _try_load_models():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        from sentence_transformers import SentenceTransformer
        tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return gen, embedder
    except Exception:
        return None, None

GEN, EMBEDDER = _try_load_models()

# ---------------- Small utilities ---------------- #
def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def safe_font(size: int = 48):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def make_gradient(w: int, h: int, start_rgb, end_rgb):
    base = Image.new("RGB", (w, h), start_rgb)
    top = Image.new("RGB", (w, h), end_rgb)
    mask = Image.linear_gradient("L").resize((w, h))
    return Image.composite(top, base, mask)

def contrast_color(rgb):
    r, g, b = rgb
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return (0, 0, 0) if yiq > 128 else (255, 255, 255)

def suggest_channel(goal: str) -> str:
    g = (goal or "").lower()
    if "aware" in g: return "IG Reels / TikTok / Shorts"
    if "engage" in g: return "Instagram Carousel / LinkedIn Post"
    if "lead" in g or "signup" in g: return "LinkedIn Lead Gen / Landing Page"
    if "sale" in g or "conversion" in g: return "Retargeting Display + Email"
    return "Social + Landing Page"

def seeded_score(*args) -> float:
    h = hashlib.sha256(("||".join([str(a) for a in args])).encode()).hexdigest()
    return round(40 + (int(h[:6], 16) % 6000) / 100.0, 1)

# ---------------- Copy engine (Lite) ---------------- #
POWER_VERBS = ["Transform", "Turn", "Make", "Scale", "Automate", "Unlock", "Elevate"]
RESULT_WORDS = {
    "Awareness": ["attention", "reach", "buzz"],
    "Engagement": ["responses", "conversations", "interaction"],
    "Signups/Leads": ["qualified leads", "signups", "pipeline"],
    "Sales/Conversions": ["conversions", "revenue", "sales"],
}
CTA_BY_GOAL = {
    "Awareness": ["See How", "Explore More", "Watch Demo"],
    "Engagement": ["Join In", "Try It Today", "Get the Playbook"],
    "Signups/Leads": ["Start Free", "Get a Demo", "Talk to Us"],
    "Sales/Conversions": ["Buy Now", "Start Now", "Claim Offer"],
}

def truncate_words(s: str, max_words: int) -> str:
    words = s.split()
    return " ".join(words[:max_words])

def polish_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    # Capitalize first char if needed
    if not s[0].isupper():
        s = s[0].upper() + s[1:]
    # Ensure terminal punctuation
    if s[-1] not in ".!?":
        s = s + "."
    return s

def compose_copy_lite(brand_desc: str, goal: str, tone: str, persona_label: str) -> Tuple[str, str, str]:
    verb = random.choice(POWER_VERBS)
    result = random.choice(RESULT_WORDS.get(goal, ["results"]))
    headline = truncate_words(f"{verb} {persona_label} rituals into {result}", 8)
    body = (
        f"For {persona_label.lower()}, ZARI blends human-centered creative with signal-driven precision — "
        f"{brand_desc.lower()} — to deliver repeatable wins you can measure."
    )
    body = polish_sentence(truncate_words(body, 28))
    cta = random.choice(CTA_BY_GOAL.get(goal, ["Learn More"]))
    return headline, body, cta

# ---------------- Clustering ---------------- #
def cluster_personas(personas: List[str], k: int):
    k = max(1, min(k, len(personas)))
    if len(personas) == 1:
        return [0], ["Segment 1: Solo"]
    if EMBEDDER is not None:
        X = EMBEDDER.encode(personas)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        lab = km.fit_predict(X)
        summaries = []
        for c in range(k):
            idx = np.where(lab == c)[0]
            centroid = km.cluster_centers_[c]
            dists = np.linalg.norm(X[idx] - centroid, axis=1)
            rep = personas[idx[np.argmin(dists)]]
            summaries.append(f"Segment {c+1}: like '{rep}'")
        return lab.tolist(), summaries
    # Fallback: simple hashing buckets
    lab = [hash(p.lower()) % k for p in personas]
    return lab, [f"Segment {i+1}: keyword-driven cluster" for i in range(k)]

# ---------------- Poster visuals ---------------- #
def draw_poster(headline: str, sub: str, cta: str, tone: str,
                size=(1080, 1350),
                template: str = "Hero",
                brand_bar: str = BRAND_BAR_DEFAULT,
                color_override: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = None) -> Image.Image:
    w, h = size
    if color_override:
        start, end = color_override
    else:
        tl = (tone or "").lower()
        if any(k in tl for k in ["premium", "lux", "elegant"]):
            start, end = ((25, 25, 30), (160, 136, 98))
        elif any(k in tl for k in ["calm", "wellness", "mindful"]):
            start, end = ((46, 204, 113), (24, 44, 37))
        elif any(k in tl for k in ["tech", "modern", "clean"]):
            start, end = ((44, 62, 80), (52, 152, 219))
        else:
            start, end = ((219, 112, 147), (44, 44, 56))

    bg = make_gradient(w, h, start, end)
    draw = ImageDraw.Draw(bg)
    text_color = contrast_color(((start[0]+end[0])//2, (start[1]+end[1])//2, (start[2]+end[2])//2))

    f_head = safe_font(size=int(0.085 * w))
    f_sub  = safe_font(size=int(0.045 * w))
    f_cta  = safe_font(size=int(0.05  * w))
    f_brand= safe_font(size=int(0.035 * w))
    pad = int(0.08 * w)

    def wrap(text, font, maxw):
        lines, cur = [], ""
        for t in text.split():
            test = (cur + " " + t).strip()
            if draw.textlength(test, font=font) <= maxw:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = t
        if cur:
            lines.append(cur)
        return lines

    if template == "Split Stripe":
        stripe_w = int(0.18 * w)
        draw.rectangle([0, 0, stripe_w, h], fill=(0, 0, 0, 40))
        x, y = pad, pad
        draw.text((x, y), headline, font=f_head, fill=text_color)
        y += int(0.11 * h)
        max_w = w - stripe_w - pad*2
        for line in wrap(sub, f_sub, max_w):
            draw.text((x, y), line, font=f_sub, fill=text_color)
            y += int(0.055 * h)
        btn_h = int(0.11*h); btn_w = int(0.58*w)
        btn_x, btn_y = x, h - pad - btn_h
    else:  # Hero
        y = pad
        draw.text((pad, y), headline, font=f_head, fill=text_color)
        y += int(0.12 * h)
        max_w = w - 2*pad
        for line in wrap(sub, f_sub, max_w):
            draw.text((pad, y), line, font=f_sub, fill=text_color)
            y += int(0.055*h)
        btn_h = int(0.11*h); btn_w = int(0.6*w)
        btn_x, btn_y = pad, h - pad - btn_h

    btn_color = (255,255,255) if sum(text_color) < 384 else (0,0,0)
    txt_color = (0,0,0) if btn_color == (255,255,255) else (255,255,255)
    draw.rounded_rectangle([btn_x, btn_y, btn_x+btn_w, btn_y+btn_h], radius=28, fill=btn_color)
    cta_w = draw.textlength(cta, font=f_cta); cta_h = f_cta.size
    draw.text((btn_x + (btn_w-cta_w)/2, btn_y + (btn_h-cta_h)/2), cta, font=f_cta, fill=txt_color)

    footer_w = draw.textlength(brand_bar, font=f_brand)
    draw.text((w - pad - footer_w, h - pad - f_brand.size - 4), brand_bar, font=f_brand, fill=text_color)
    return bg

# ---------------- MoviePy availability ---------------- #
@st.cache_resource(show_spinner=False)
def _has_moviepy():
    try:
        from moviepy.editor import ImageClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips, vfx
        return True
    except Exception:
        return False

HAS_MOVIEPY = _has_moviepy()

# ---------------- Video makers ---------------- #
def make_short_video_from_images(intro: Image.Image, mid: Image.Image, outro: Image.Image,
                                 fps: int = 24, duration_each: float = 2.0, out_size=(720,1280)) -> bytes:
    """3-scene slideshow MP4 with fades."""
    from moviepy.editor import ImageClip, concatenate_videoclips, vfx
    def to_clip(pil_img):
        if pil_img.size != out_size:
            pil_img = pil_img.resize(out_size, Image.LANCZOS)
        return ImageClip(np.array(pil_img)).set_duration(duration_each)
    c1 = to_clip(intro).fx(vfx.fadein, 0.3)
    c2 = to_clip(mid).fx(vfx.fadein, 0.3)
    c3 = to_clip(outro).fx(vfx.fadein, 0.3).fx(vfx.fadeout, 0.3)
    clip = concatenate_videoclips([c1, c2, c3], method="compose")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out = tmp.name
    clip.write_videofile(out, fps=fps, audio=False, codec="libx264",
                         preset="ultrafast", bitrate="1500k", logger=None)
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data

def _overlay_img(w, h, text_lines, button_text=None, alpha=0.12) -> Image.Image:
    im = Image.new("RGBA", (w, h), (0,0,0, int(alpha*255)))
    d = ImageDraw.Draw(im)
    f_head = safe_font(int(0.08*w))
    f_body = safe_font(int(0.045*w))
    f_cta  = safe_font(int(0.05*w))
    pad = int(0.07*w); y = int(0.15*h)
    color = (255,255,255,255)
    for i, line in enumerate(text_lines):
        d.text((pad, y), line, font=(f_head if i==0 else f_body), fill=color)
        y += int(0.09*h if i==0 else 0.06*h)
    if button_text:
        btn_w = int(0.6*w); btn_h = int(0.11*h); btn_x = pad; btn_y = h - int(0.18*h)
        d.rounded_rectangle([btn_x,btn_y,btn_x+btn_w,btn_y+btn_h], radius=28, fill=(255,255,255,235))
        cta_w = d.textlength(button_text, font=f_cta); cta_h = f_cta.size
        d.text((btn_x+(btn_w-cta_w)/2, btn_y+(btn_h-cta_h)/2), button_text, font=f_cta, fill=(0,0,0,255))
    return im

def brand_over_base_video(base_bytes: bytes, headline: str, body: str, cta: str,
                          out_size=(720,1280), fps=24) -> bytes:
    """Overlay headline/body/CTA on an uploaded MP4."""
    from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, vfx
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(base_bytes); base_path = tmp.name
    base = VideoFileClip(base_path).resize(out_size)
    dur = max(6.0, min(base.duration, 12.0))
    seg = dur / 3.0
    intro = _overlay_img(*out_size, [headline], None, alpha=0.15)
    mid   = _overlay_img(*out_size, [headline, body], None, alpha=0.12)
    outro = _overlay_img(*out_size, [headline], cta, alpha=0.12)
    def ov(pil, start):
        return ImageClip(np.array(pil)).set_start(start).set_duration(seg).fx(vfx.fadein,0.3).fx(vfx.fadeout,0.3)
    comp = CompositeVideoClip([base, ov(intro,0), ov(mid,seg), ov(outro,seg*2)])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out:
        out_path = out.name
    comp.write_videofile(out_path, fps=fps, audio=False, codec="libx264",
                         preset="ultrafast", bitrate="2000k", logger=None)
    with open(out_path, "rb") as f:
        data = f.read()
    for p in [base_path, out_path]:
        try: os.remove(p)
        except Exception: pass
    return data

# ---------------- Packaging ---------------- #
def package_zip(artifacts: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("copy.json", json.dumps(artifacts["copy"], indent=2))
        for item in artifacts["images"]:
            img: Image.Image = item["image"]
            b = io.BytesIO(); img.save(b, format="PNG"); b.seek(0)
            zf.writestr(item["path"], b.read())
        for v in artifacts.get("videos", []):
            zf.writestr(v["path"], v["data"])
        zf.writestr("README.txt", "Ritual Creative Lab — PNG posters, MP4 videos, and copy.json.")
    buf.seek(0)
    return buf.read()

# ---------------- UI ---------------- #
def header_bar():
    st.markdown(
        """
        <div style="padding:10px 16px;border-radius:12px;background:#0f172a;color:#e2e8f0;
             display:flex;gap:12px;justify-content:space-between;align-items:center;">
            <div style="font-size:18px;font-weight:700;">Ritual Creative Lab — POC (Powered by ZARI)</div>
            <div style="opacity:0.85;">Free • Local models • No API keys</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def load_demo():
    st.session_state.brand_desc = "ZARI builds AI operators for founders and lean teams—turning repeatable work into reliable rituals that run themselves, stay on-brand, and deliver measurable ROI."
    st.session_state.goal = "Engagement"
    st.session_state.tone = "Bold, modern, human-centered"
    st.session_state.personas_text = "\n".join([
        "Seed-stage SaaS founders (2–20 employees)",
        "Shopify/DTC operators (>$50k/mo)",
        "Service business owners who book appointments",
        "Real estate team leads",
        "Healthcare clinic managers",
        "Course creators & coaches",
    ])

def main():
    header_bar()
    with st.sidebar:
        st.subheader("Brand & Campaign")
        if "brand_desc" not in st.session_state: st.session_state.brand_desc = ""
        if "goal" not in st.session_state: st.session_state.goal = "Engagement"
        if "tone" not in st.session_state: st.session_state.tone = "Bold, modern, human-centered"
        if "personas_text" not in st.session_state: st.session_state.personas_text = ""

        c1, c2 = st.columns(2)
        if c1.button("Load Demo Campaign"): load_demo()
        copy_mode = c2.selectbox("Copy mode", ["Lite Templates (recommended)", "FLAN-small (experimental)"])

        brand_desc = st.text_area("Brand description (who/why/ritual):", key="brand_desc", height=110)
        goal = st.selectbox("Primary goal", ["Awareness", "Engagement", "Signups/Leads", "Sales/Conversions"], key="goal")
        tone = st.text_input("Voice / Tone", key="tone")

        st.markdown("---")
        st.subheader("Personas (one per line)")
        personas_text = st.text_area("Examples: 'Fitness-curious millennials', 'CTOs at seed-stage SaaS', etc.", key="personas_text", height=140)
        k_clusters = st.slider("Number of persona segments", 1, 6, 3)
        n_variants = st.slider("Variants per segment", 1, 3, 2)

        st.markdown("---")
        st.subheader("Brand Kit (optional)")
        col = st.columns(2)
        primary_hex = col[0].color_picker("Primary", "#7c3aed")
        secondary_hex = col[1].color_picker("Secondary", "#111827")
        brand_bar_text = st.text_input("Footer brand bar", BRAND_BAR_DEFAULT)
        template_style = st.selectbox("Poster template", ["Hero", "Split Stripe"])

        st.markdown("---")
        st.subheader("Video (optional)")
        gen_videos = st.toggle("Generate videos (MP4)", value=True)
        video_mode = st.radio("Source", ["Auto (from posters)", "Upload base AI clip"], horizontal=True)
        uploaded_base = None
        if video_mode == "Upload base AI clip":
            uploaded_base = st.file_uploader("Upload MP4", type=["mp4"],
                                             help="Use any short AI human clip from Colab or elsewhere.")
        video_res = st.selectbox("Resolution", ["720x1280 (fast)", "1080x1920 (slower)"])
        res_map = {"720x1280 (fast)": (720,1280), "1080x1920 (slower)": (1080,1920)}

        st.markdown("---")
        with st.expander("Integrations (preview)"):
            st.button("Connect Google Ads (coming soon)", disabled=True, use_container_width=True)
            st.button("Connect Meta Business Suite (coming soon)", disabled=True, use_container_width=True)
            st.button("Export to HubSpot / Salesforce (coming soon)", disabled=True, use_container_width=True)

    personas = [p.strip() for p in personas_text.splitlines() if p.strip()]
    tabs = st.tabs(["Plan", "Generate", "Review & Download"])

    with tabs[0]:
        st.markdown("### Creative Plan")
        st.write(
            "- Cluster personas → generate tight ad copy → auto-create premium posters.\n"
            "- **Videos:** either an auto slideshow from posters or a branded overlay on your uploaded MP4.\n"
            "- Export PNGs, MP4s, and copy.json as a ZIP."
        )
        if brand_desc: st.info(f"Brand Summary: {brand_desc}")
        if personas:
            st.success(f"{len(personas)} persona(s) provided.")
            st.markdown("#### Simulated Impact Preview")
            df_prev = pd.DataFrame({
                "Persona": personas[:8],
                "Predicted Engagement Lift (%)": [seeded_score(brand_desc, p, goal, tone) for p in personas[:8]],
            }).set_index("Persona")
            st.bar_chart(df_prev)

    with tabs[1]:
        st.markdown("### Generate Ad Packages")
        if st.button("Run Generator", type="primary", use_container_width=True):
            if not brand_desc or not personas:
                st.error("Please provide a brand description and at least one persona.")
                st.stop()

            labels, summaries = cluster_personas(personas, k_clusters)
            st.session_state["segments"] = []
            now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            colors = (hex_to_rgb(primary_hex), hex_to_rgb(secondary_hex))
            artifacts_videos = []
            images_for_zip = []

            for seg_idx, seg_title in enumerate(summaries):
                seg_personas = [p for p, lab in zip(personas, labels) if lab == seg_idx]
                if not seg_personas: continue
                rep_label = seg_personas[0]

                st.write(f"#### {seg_title}")
                st.caption(f"Personas in this segment: {', '.join(seg_personas)}")
                seg_outputs = {"segment": seg_title, "variants": []}

                for v in range(n_variants):
                    # Copy
                    if GEN is None or copy_mode.startswith("Lite"):
                        headline, body, cta = compose_copy_lite(brand_desc, goal, tone, rep_label)
                    else:
                        try:
                            prompt = f"Write a short ad for {rep_label}. Goal: {goal}. Tone: {tone}. Headline<=8w; Body<=30w; CTA<=6w. Format: HEADLINE:, BODY:, CTA:"
                            raw = GEN(prompt, max_length=128, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]
                            m_h = re.search(r"HEADLINE:\s*(.+)", raw, re.I)
                            m_b = re.search(r"BODY:\s*(.+)", raw, re.I)
                            m_c = re.search(r"CTA:\s*(.+)", raw, re.I)
                            headline = truncate_words((m_h.group(1) if m_h else raw[:64]).strip(), 8)
                            body = polish_sentence(truncate_words((m_b.group(1) if m_b else raw).strip(), 30))
                            cta = truncate_words((m_c.group(1) if m_c else "Learn More").strip(), 6)
                        except Exception:
                            headline, body, cta = compose_copy_lite(brand_desc, goal, tone, rep_label)

                    channel = suggest_channel(goal)

                    # Posters (for preview and ZIP)
                    intro = draw_poster(headline, "", cta, tone, size=(1080,1350),
                                        template=template_style, brand_bar=brand_bar_text, color_override=colors)
                    mid   = draw_poster(headline, body, cta, tone, size=(1080,1350),
                                        template=template_style, brand_bar=brand_bar_text, color_override=colors)
                    c1, c2 = st.columns(2)
                    c1.image(intro, caption=f"Variant {v+1} Visual A", use_column_width=True)
                    c2.image(mid,   caption=f"Variant {v+1} Visual B", use_column_width=True)
                    st.caption(f"Predicted Engagement Lift: **{seeded_score(brand_desc, seg_title, goal, tone, v)}%** • Suggested: **{channel}**")

                    # Video
                    video_bytes = None
                    if gen_videos and HAS_MOVIEPY:
                        try:
                            if video_mode == "Upload base AI clip" and uploaded_base is not None:
                                base_bytes = uploaded_base.read()
                                video_bytes = brand_over_base_video(base_bytes, headline, body, cta,
                                                                    out_size=res_map[video_res], fps=24)
                            else:
                                video_bytes = make_short_video_from_images(
                                    intro, mid, mid, fps=24, duration_each=2.0, out_size=res_map[video_res]
                                )
                            st.video(video_bytes)
                            artifacts_videos.append({"path": f"videos/segment{seg_idx+1}_variant{v+1}.mp4", "data": video_bytes})
                        except Exception as e:
                            st.warning(f"Video render failed: {e}")
                    elif gen_videos and not HAS_MOVIEPY:
                        st.warning("MoviePy/ffmpeg not available yet — skipping video this run.")

                    seg_outputs["variants"].append({
                        "headline": headline, "body": body, "cta": cta,
                        "channel": channel, "personas": seg_personas,
                        "video": bool(video_bytes)
                    })

                    # Add posters to ZIP
                    for ab, img in zip(["A","B"], [intro, mid]):
                        images_for_zip.append({
                            "image": img,
                            "path": f"images/segment{seg_idx+1}_variant{v+1}_{ab}.png"
                        })

                st.session_state["segments"].append(seg_outputs)

            copy_payload = {
                "brand": brand_desc, "goal": goal, "tone": tone,
                "segments": st.session_state["segments"],
                "meta": {"generated_at_utc": datetime.utcnow().isoformat()+"Z",
                         "videos_enabled": bool(gen_videos and HAS_MOVIEPY)}
            }
            st.session_state["artifacts"] = {
                "copy": copy_payload,
                "images": images_for_zip,
                "videos": artifacts_videos,
                "tag": now_tag
            }
            st.success("Generation complete. Review & download in the next tab.")

    with tabs[2]:
        st.markdown("### Review & Download")
        if "artifacts" not in st.session_state:
            st.info("Run the generator first.")
        else:
            for seg in st.session_state["artifacts"]["copy"]["segments"]:
                st.write(f"**{seg['segment']}**")
                for idx, v in enumerate(seg["variants"], start=1):
                    st.write(
                        f"- **V{idx}** — **{v['headline']}**  \n"
                        f"  {v['body']}  \n"
                        f"  _CTA: {v['cta']}_  \n"
                        f"  _Channel: {v['channel']}_  \n"
                        f"  _Video:_ {'Yes' if v.get('video') else 'No'}"
                    )

            zip_bytes = package_zip(st.session_state["artifacts"])
            name = f"ritual_ads_poc_{st.session_state['artifacts']['tag']}.zip"
            st.download_button("⬇️ Download ZIP (PNG + MP4 + copy.json)",
                               data=zip_bytes, file_name=name, mime="application/zip",
                               use_container_width=True)
            st.caption("All assets are generated locally on CPU.")

if __name__ == "__main__":
    main()
