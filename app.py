# Ritual Creative Lab — POC (Powered by ZARI)
# Video upgrade: clean copy, premium posters, optional short MP4 videos per variant.
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

# ---------------- Optional models (kept for experimentation) ---------------- #
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
HAS_MODELS = GEN is not None and EMBEDDER is not None
random.seed(RANDOM_SEED)

# ---------------- Utilities ---------------- #
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
    if "awareness" in g: return "IG Reels / TikTok / YouTube Shorts"
    if "engagement" in g: return "Instagram Carousel / LinkedIn Post"
    if "signup" in g or "lead" in g: return "LinkedIn Lead Gen / Landing Page"
    if "sale" in g or "conversion" in g: return "Retargeting Display + Email"
    return "Social + Landing Page"

def seeded_score(*args) -> float:
    h = hashlib.sha256(("||".join([str(a) for a in args])).encode()).hexdigest()
    val = int(h[:6], 16)
    return round(40 + (val % 6000) / 100.0, 1)  # 40.0–100.0

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
    if not s: return s
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?": s += "."
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

def gen_text(gen, prompt: str, max_tokens: int = 128) -> str:
    out = gen(prompt, max_length=max_tokens, do_sample=True, top_p=0.9, temperature=0.7, num_return_sequences=1)
    return out[0]["generated_text"].strip()

def make_copy_prompt(brand_desc: str, goal: str, tone: str, persona_label: str) -> str:
    return f"""
You are a senior creative at a human-centered AI advertising studio.
Brand: {brand_desc}
Goal: {goal}
Tone: {tone}
Persona: {persona_label}

Generate:
1) HEADLINE (<=8 words)
2) BODY (<=30 words)
3) CTA (<=6 words)
Format exactly:
HEADLINE: <headline>
BODY: <body>
CTA: <cta>
""".strip()

HEAD_RE = re.compile(r"^HEADLINE:\s*(.+)$", re.I|re.M)
BODY_RE = re.compile(r"^BODY:\s*(.+)$", re.I|re.M)
CTA_RE  = re.compile(r"^CTA:\s*(.+)$", re.I|re.M)

def parse_copy(text: str) -> Tuple[str, str, str]:
    h = HEAD_RE.search(text); b = BODY_RE.search(text); c = CTA_RE.search(text)
    headline = truncate_words((h.group(1).strip() if h else text[:70]), 8)
    body = polish_sentence(truncate_words((b.group(1).strip() if b else text), 30))
    cta = truncate_words((c.group(1).strip() if c else "Learn More"), 6)
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
    # fallback: lexical hash buckets
    lab = [(hash(p.lower()) % k) for p in personas]
    summaries = [f"Segment {i+1}: keyword-driven cluster" for i in range(k)]
    return lab, summaries

# ---------------- Visuals (premium posters) ---------------- #
def draw_poster(headline: str, sub: str, cta: str, tone: str,
                size=(1080, 1350),
                template: str = "Hero",
                brand_bar: str = BRAND_BAR_DEFAULT,
                color_override: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = None) -> Image.Image:
    w, h = size
    if color_override:
        start, end = color_override
    else:
        if any(k in (tone or "").lower() for k in ["premium","lux","elegant"]):
            start, end = ((25,25,30), (160,136,98))
        elif any(k in (tone or "").lower() for k in ["calm","wellness","mindful"]):
            start, end = ((46,204,113), (24,44,37))
        elif any(k in (tone or "").lower() for k in ["tech","modern","clean"]):
            start, end = ((44,62,80), (52,152,219))
        else:
            start, end = ((219,112,147), (44,44,56))

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
            if draw.textlength(test, font=font) <= maxw: cur = test
            else: lines.append(cur); cur = t
        if cur: lines.append(cur)
        return lines

    if template == "Split Stripe":
        stripe_w = int(0.18 * w)
        draw.rectangle([0,0,stripe_w,h], fill=(0,0,0,40))
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
            draw.text((pad, y), line, font=f_sub, fill=text_color); y += int(0.055*h)
        btn_h = int(0.11*h); btn_w = int(0.6*w)
        btn_x, btn_y = pad, h - pad - btn_h

    btn_color = (255,255,255) if sum(text_color) < 384 else (0,0,0)
    txt_color = (0,0,0) if btn_color == (255,255,255) else (255,255,255)
    draw.rounded_rectangle([btn_x, btn_y, btn_x+btn_w, btn_y+btn_h], radius=28, fill=btn_color)
    cta_w = draw.textlength(cta, font=f_cta); cta_h = f_cta.size
    draw.text((btn_x + (btn_w-cta_w)/2, btn_y + (btn_h-cta_h)/2), cta, font=f_cta, fill=txt_color)

    footer = brand_bar
    footer_w = draw.textlength(footer, font=f_brand)
    draw.text((w - pad - footer_w, h - pad - f_brand.size - 4), footer, font=f_brand, fill=text_color)
    return bg

# ---------------- Video (MoviePy) ---------------- #
@st.cache_resource(show_spinner=False)
def _try_moviepy():
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips, vfx
        return True
    except Exception:
        return False

HAS_MOVIEPY = _try_moviepy()

def make_short_video_mp4(img_intro: Image.Image, img_mid: Image.Image, img_outro: Image.Image,
                         fps: int = 24, duration_each: float = 2.0, out_size=(720, 1280)) -> bytes:
    """
    Build a simple 3-scene vertical MP4 with crossfades from PIL images.
    """
    from moviepy.editor import ImageClip, concatenate_videoclips, vfx

    def to_clip(pil_img):
        # Ensure target size
        if pil_img.size != out_size:
            pil_img = pil_img.resize(out_size, Image.LANCZOS)
        return ImageClip(np.array(pil_img)).set_duration(duration_each)

    c1 = to_clip(img_intro).fx(vfx.fadein, 0.4)
    c2 = to_clip(img_mid).fx(vfx.fadein, 0.4)
    c3 = to_clip(img_outro).fx(vfx.fadein, 0.4).fx(vfx.fadeout, 0.4)
    clip = concatenate_videoclips([c1, c2, c3], method="compose")

    # write to temp path, then return bytes
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    clip.write_videofile(tmp_path, fps=fps, audio=False, codec="libx264",
                         preset="ultrafast", bitrate="1500k", logger=None)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data

# ---------------- Packaging ---------------- #
def package_zip(artifacts: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("copy.json", json.dumps(artifacts["copy"], indent=2))
        for item in artifacts["images"]:
            img: Image.Image = item["image"]
            path = item["path"]
            b = io.BytesIO(); img.save(b, format="PNG"); b.seek(0)
            zf.writestr(path, b.read())
        for v in artifacts.get("videos", []):
            zf.writestr(v["path"], v["data"])
        zf.writestr("README.txt", "Ritual Creative Lab — generated PNG posters, MP4 videos (if enabled), and copy.json.")
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
        "Fitness-curious millennials",
        "CTOs at seed-stage SaaS",
        "Wellness-focused new moms",
        "Creators launching digital courses",
        "Eco-conscious urban professionals",
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
        k_clusters = st.slider("Number of persona segments", 1, 6, 2)
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
        gen_videos = st.toggle("Generate videos (MP4)", value=False,
                               help="Creates a short vertical MP4 per variant (3 scenes with fades).")
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
            "- Optional: generate short MP4 videos per variant (3 scenes with fades).\n"
            "- Outputs: headline, body, CTA, channel, PNGs (+ MP4 if enabled) and a ZIP export."
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
                st.error("Please provide a brand description and at least one persona in the sidebar.")
                st.stop()

            labels, summaries = cluster_personas(personas, k_clusters)
            st.session_state["segments"] = []
            now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            colors = (hex_to_rgb(primary_hex), hex_to_rgb(secondary_hex))

            artifacts_videos = []  # accumulate for ZIP

            for seg_idx, seg_title in enumerate(summaries):
                seg_personas = [p for p, lab in zip(personas, labels) if lab == seg_idx]
                if not seg_personas: continue
                rep_label = seg_personas[0]  # representative persona for cleaner headlines

                st.write(f"#### {seg_title}")
                st.caption(f"Personas in this segment: {', '.join(seg_personas)}")

                seg_outputs = {"segment": seg_title, "variants": []}
                for v in range(n_variants):
                    if copy_mode.startswith("Lite") or GEN is None:
                        headline, body, cta = compose_copy_lite(brand_desc, goal, tone, rep_label)
                    else:
                        prompt = make_copy_prompt(brand_desc, goal, tone, rep_label)
                        try:
                            raw = gen_text(GEN, prompt, max_tokens=128)
                            headline, body, cta = parse_copy(raw)
                        except Exception:
                            headline, body, cta = compose_copy_lite(brand_desc, goal, tone, rep_label)

                    channel = suggest_channel(goal)
                    # posters
                    intro = draw_poster(headline, "", cta, tone, size=(1080, 1350),
                                        template=template_style, brand_bar=brand_bar_text, color_override=colors)
                    mid   = draw_poster(headline, body, cta, tone, size=(1080, 1350),
                                        template=template_style, brand_bar=brand_bar_text, color_override=colors)
                    outro = draw_poster(headline, body, cta, tone, size=(1080, 1350),
                                        template=template_style, brand_bar=brand_bar_text, color_override=colors)

                    cols = st.columns(2)
                    cols[0].image(intro, caption=f"Variant {v+1} Visual A", use_column_width=True)
                    cols[1].image(mid,   caption=f"Variant {v+1} Visual B", use_column_width=True)
                    st.caption(f"Predicted Engagement Lift: **{seeded_score(brand_desc, seg_title, goal, tone, v)}%**  •  Suggested: **{channel}**")

                    # optional video
                    video_bytes = None
                    if gen_videos:
                        if not HAS_MOVIEPY:
                            st.warning("MoviePy/ffmpeg unavailable — skipping video. (It will auto-install on next deploy.)")
                        else:
                            try:
                                video_bytes = make_short_video_mp4(intro, mid, outro, fps=24,
                                                                   duration_each=2.0,
                                                                   out_size=res_map[video_res])
                                st.video(video_bytes)
                                artifacts_videos.append({
                                    "path": f"videos/segment{seg_idx+1}_variant{v+1}.mp4",
                                    "data": video_bytes
                                })
                            except Exception as e:
                                st.warning(f"Video render failed: {e}")

                    seg_outputs["variants"].append({
                        "headline": headline, "body": body, "cta": cta,
                        "channel": channel, "personas": seg_personas,
                        "video": bool(video_bytes)
                    })

                st.session_state["segments"].append(seg_outputs)

            # Prepare artifacts for ZIP
            images_for_zip = []
            for s_i, seg in enumerate(st.session_state["segments"]):
                for v_i, variant in enumerate(seg["variants"]):
                    for ab, img in zip(["A","B"], [
                        draw_poster(variant["headline"], "", variant["cta"], tone,
                                    size=(1080, 1350), template=template_style,
                                    brand_bar=brand_bar_text, color_override=colors),
                        draw_poster(variant["headline"], variant["body"], variant["cta"], tone,
                                    size=(1080, 1350), template=template_style,
                                    brand_bar=brand_bar_text, color_override=colors),
                    ]):
                        path = f"images/segment{s_i+1}_variant{v_i+1}_{ab}.png"
                        images_for_zip.append({"image": img, "path": path})

            copy_payload = {
                "brand": brand_desc, "goal": goal, "tone": tone,
                "segments": st.session_state["segments"],
                "meta": {"generated_at_utc": datetime.utcnow().isoformat() + "Z",
                         "copy_mode": "lite" if copy_mode.startswith("Lite") or GEN is None else "flan-small",
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
            st.caption("All assets are generated locally. Modify tone/personas or brand kit, re-run, and download again.")

if __name__ == "__main__":
    main()
