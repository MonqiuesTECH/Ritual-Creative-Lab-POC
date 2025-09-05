# Ritual Creative Lab — POC (Powered by ZARI)
# Free, CPU-only Streamlit app. No API keys. No paid services.

import hashlib
import io
import json
import math
import random
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

APP_TITLE = "Ritual Creative Lab — POC (Powered by ZARI)"
BRAND_BAR = "Ritual Ads • Powered by ZARI"
RANDOM_SEED = 42  # deterministic demo behavior

# --- Streamlit page config MUST be the first Streamlit call ---
st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")


# -------------------------------------------------------------------
# Model loading with graceful fallback (Lite Mode)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def _try_load_models():
    """
    Try loading transformers + sentence-transformers.
    If anything fails (e.g., memory/time), return (None, None) and the app will switch to Lite Mode.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        from sentence_transformers import SentenceTransformer

        tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return gen, embedder
    except Exception as e:
        return None, None


GEN, EMBEDDER = _try_load_models()
HAS_MODELS = GEN is not None and EMBEDDER is not None


def gen_text(gen, prompt: str, max_tokens: int = 128) -> str:
    """Generate text from local T5 model with conservative sampling."""
    out = gen(
        prompt,
        max_length=max_tokens,
        do_sample=True,
        top_p=0.92,
        temperature=0.8,
        num_return_sequences=1,
    )
    return out[0]["generated_text"].strip()


def gen_text_lite(brand_desc: str, goal: str, tone: str, persona: str) -> Tuple[str, str, str]:
    """
    Lite Mode templated generation (no ML). Provides clean, believable copy if models are unavailable.
    """
    random.seed(hash((brand_desc, goal, tone, persona)) % (2**32))
    # Headline options
    h_templates = [
        "Make It Ritual",
        "Turn Moments Into Meaning",
        "Where Story Meets Action",
        "Designed to Be Remembered",
        "Rituals That Move People",
        "Crafted For Connection",
        "Heart + Data = Impact",
    ]
    headline = random.choice(h_templates)
    # Body
    body = (
        f"For {persona.lower()}, we connect {brand_desc.lower()} with {goal.lower()} outcomes — "
        f"human-centered creative guided by signals, delivered with modern precision."
    )
    # CTA
    ctas = ["See How", "Start Your Ritual", "Try It Today", "Get the Playbook", "Learn More"]
    cta = random.choice(ctas)
    return headline, body, cta


# -------------------------------------------------------------------
# Visuals
# -------------------------------------------------------------------
def safe_font(size: int = 48):
    """Try a common TTF; fallback to PIL default if unavailable."""
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


def pick_palette(tone: str):
    t = (tone or "").lower()
    if any(k in t for k in ["bold", "energetic", "disrupt"]):
        return ((255, 94, 98), (36, 36, 36))    # coral -> charcoal
    if any(k in t for k in ["premium", "lux", "elegant"]):
        return ((25, 25, 30), (160, 136, 98))   # midnight -> warm gold
    if any(k in t for k in ["calm", "wellness", "mindful"]):
        return ((46, 204, 113), (24, 44, 37))   # jade -> deep green
    if any(k in t for k in ["tech", "modern", "clean"]):
        return ((44, 62, 80), (52, 152, 219))   # slate -> azure
    return ((219, 112, 147), (44, 44, 56))      # rose -> deep slate


def draw_poster(headline: str, sub: str, cta: str, tone: str, size=(1080, 1350)) -> Image.Image:
    w, h = size
    start, end = pick_palette(tone)
    bg = make_gradient(w, h, start, end)
    draw = ImageDraw.Draw(bg)

    pad = int(0.08 * w)
    f_head = safe_font(size=int(0.08 * w))
    f_sub = safe_font(size=int(0.045 * w))
    f_cta = safe_font(size=int(0.05 * w))
    f_brand = safe_font(size=int(0.035 * w))

    text_color = contrast_color(((start[0] + end[0]) // 2, (start[1] + end[1]) // 2, (start[2] + end[2]) // 2))

    # Headline
    y = pad
    draw.text((pad, y), headline, font=f_head, fill=text_color)
    y += int(0.12 * h)

    # Subcopy wrap
    def wrap(text, font, max_width):
        words = text.split()
        lines, cur = [], ""
        for w_ in words:
            test = (cur + " " + w_).strip()
            if draw.textlength(test, font=font) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w_
        if cur:
            lines.append(cur)
        return lines

    max_w = w - 2 * pad
    for line in wrap(sub, f_sub, max_w):
        draw.text((pad, y), line, font=f_sub, fill=text_color)
        y += int(0.055 * h)

    # CTA button
    btn_h = int(0.11 * h)
    btn_w = int(0.6 * w)
    btn_x = pad
    btn_y = h - pad - btn_h
    btn_color = (255, 255, 255) if sum(text_color) < 384 else (0, 0, 0)
    txt_color = (0, 0, 0) if btn_color == (255, 255, 255) else (255, 255, 255)

    draw.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], radius=28, fill=btn_color)
    cta_w = draw.textlength(cta, font=f_cta)
    cta_h = f_cta.size
    draw.text((btn_x + (btn_w - cta_w) / 2, btn_y + (btn_h - cta_h) / 2), cta, font=f_cta, fill=txt_color)

    # Brand footer
    footer_w = draw.textlength(BRAND_BAR, font=f_brand)
    draw.text((w - pad - footer_w, h - pad - f_brand.size - 4), BRAND_BAR, font=f_brand, fill=text_color)
    return bg


# -------------------------------------------------------------------
# Copy generation prompts + parsing
# -------------------------------------------------------------------
def make_copy_prompt(brand_desc: str, goal: str, tone: str, persona: str) -> str:
    return f"""
You are a senior creative at a human-centered AI advertising studio.
Brand: {brand_desc}
Goal: {goal}
Tone: {tone}
Persona: {persona}

Generate:
1) A high-impact ad headline (max 8 words).
2) One-sentence body copy (max 30 words) connecting ritual, emotion, and benefit.
3) A direct, invitational CTA (max 6 words).
Format:
HEADLINE: ...
BODY: ...
CTA: ...
""".strip()


def parse_copy(text: str) -> Tuple[str, str, str]:
    lines = text.splitlines()
    get = lambda key: next((l.split(":", 1)[1].strip() for l in lines if l.strip().lower().startswith(f"{key}:")), "")
    h = get("headline") or text[:70]
    b = get("body") or text
    c = get("cta") or "Learn More"
    return h, b, c


# -------------------------------------------------------------------
# Persona clustering
# -------------------------------------------------------------------
def cluster_personas(personas: List[str], k: int):
    """
    Use embeddings if available; otherwise fall back to lexical hashing clusters.
    Returns labels and human-readable segment summaries.
    """
    k = max(1, min(k, len(personas)))
    if HAS_MODELS:
        X = EMBEDDER.encode(personas)
        if len(personas) == 1:
            return [0], ["Segment 1: Solo"]
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
    else:
        # Lite clustering: hash tokens to buckets
        lab = []
        for p in personas:
            bucket = hash(p.lower()) % k
            lab.append(bucket)
        summaries = [f"Segment {i+1}: keyword-driven cluster" for i in range(k)]
        return lab, summaries


# -------------------------------------------------------------------
# Helpers: channels, analytics, packaging
# -------------------------------------------------------------------
def suggest_channel(goal: str) -> str:
    g = (goal or "").lower()
    if "awareness" in g:
        return "IG Reels / TikTok / YouTube Shorts"
    if "engagement" in g:
        return "Instagram Carousel / LinkedIn Post"
    if "signups" in g or "leads" in g:
        return "LinkedIn Lead Gen / Landing Page"
    if "sales" in g or "conversions" in g:
        return "Retargeting Display + Email"
    return "Social + Landing Page"


def seeded_score(*args) -> float:
    """
    Deterministic 0-100 score based on hash of inputs — looks like analytics,
    useful for demo without real data.
    """
    h = hashlib.sha256(("||".join([str(a) for a in args])).encode()).hexdigest()
    val = int(h[:6], 16)  # take first 3 bytes
    return round(40 + (val % 6000) / 100.0, 1)  # 40.0 to ~100.0


def package_zip(artifacts: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("copy.json", json.dumps(artifacts["copy"], indent=2))
        for item in artifacts["images"]:
            img: Image.Image = item["image"]
            path = item["path"]
            b = io.BytesIO()
            img.save(b, format="PNG")
            b.seek(0)
            zf.writestr(path, b.read())
    buf.seek(0)
    return buf.read()


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
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
    st.session_state.brand_desc = "We blend timeless ritual with human-centered AI to move hearts and drive results."
    st.session_state.goal = "Engagement"
    st.session_state.tone = "Bold, modern, human-centered"
    st.session_state.personas_text = "\n".join(
        [
            "Fitness-curious millennials",
            "CTOs at seed-stage SaaS",
            "Wellness-focused new moms",
            "Creators launching digital courses",
            "Eco-conscious urban professionals",
        ]
    )


def main():
    random.seed(RANDOM_SEED)
    header_bar()

    with st.sidebar:
        st.subheader("Brand & Campaign")
        if "brand_desc" not in st.session_state:
            st.session_state.brand_desc = ""
        if "goal" not in st.session_state:
            st.session_state.goal = "Engagement"
        if "tone" not in st.session_state:
            st.session_state.tone = "Bold, modern, human-centered"
        if "personas_text" not in st.session_state:
            st.session_state.personas_text = ""

        colb1, colb2 = st.columns(2)
        if colb1.button("Load Demo Campaign"):
            load_demo()
        lite_mode = colb2.toggle("Lite Mode", value=not HAS_MODELS, help="Use fast templated generation if models can't load.")

        brand_desc = st.text_area("Brand description (who/why/ritual):", key="brand_desc", height=120)
        goal = st.selectbox("Primary goal", ["Awareness", "Engagement", "Signups/Leads", "Sales/Conversions"], key="goal")
        tone = st.text_input("Voice / Tone", key="tone")

        st.markdown("---")
        st.subheader("Personas (one per line)")
        personas_text = st.text_area(
            "Examples: 'Fitness-curious millennials', 'CTOs at seed-stage SaaS', etc.",
            key="personas_text",
            height=140,
        )
        k_clusters = st.slider("Number of persona segments", min_value=1, max_value=6, value=2, help="How many audience clusters to generate.")

        st.markdown("---")
        n_variants = st.slider("Variants per segment", 1, 3, 2)
        st.caption("Branding: Ritual Ads • Powered by ZARI")

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
            "- Cluster personas → generate copy per segment → auto-create poster visuals.\n"
            "- Everything runs **locally** with open-source models on CPU.\n"
            "- Outputs: **headline, body, CTA**, suggested **channel**, and **2 PNG visuals** per variant.\n"
            "- Download a **ZIP** (PNG + copy.json) from the final tab."
        )
        if brand_desc:
            st.info(f"Brand Summary: {brand_desc}")
        if personas:
            st.success(f"{len(personas)} persona(s) provided.")

        # Analytics Preview (simulated)
        if personas:
            st.markdown("#### Simulated Impact Preview")
            df_prev = pd.DataFrame({
                "Persona": personas[:8],  # show up to 8 for readability
                "Predicted Engagement Lift (%)": [seeded_score(brand_desc, p, goal, tone) for p in personas[:8]],
            })
            st.bar_chart(df_prev.set_index("Persona"))

    with tabs[1]:
        st.markdown("### Generate Ad Packages")
        if st.button("Run Generator", type="primary", use_container_width=True):
            if not brand_desc or not personas:
                st.error("Please provide a brand description and at least one persona in the sidebar.")
                st.stop()

            labels, summaries = cluster_personas(personas, k_clusters)
            st.session_state["segments"] = []
            now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            for seg_idx, seg_title in enumerate(summaries):
                seg_personas = [p for p, lab in zip(personas, labels) if lab == seg_idx]
                if not seg_personas:
                    continue

                st.write(f"#### {seg_title}")
                st.caption(f"Personas in this segment: {', '.join(seg_personas)}")

                seg_outputs = {"segment": seg_title, "variants": []}
                for v in range(n_variants):
                    persona_hint = ", ".join(seg_personas[:3])
                    if not lite_mode and HAS_MODELS:
                        prompt = make_copy_prompt(brand_desc, goal, tone, f"{seg_title}. People like: {persona_hint}.")
                        try:
                            from transformers import Pipeline  # type: ignore
                            raw = gen_text(GEN, prompt, max_tokens=128)
                            headline, body, cta = parse_copy(raw)
                        except Exception:
                            headline, body, cta = gen_text_lite(brand_desc, goal, tone, seg_title)
                    else:
                        headline, body, cta = gen_text_lite(brand_desc, goal, tone, seg_title)

                    channel = suggest_channel(goal)
                    posters = [draw_poster(headline, body, cta, tone, size=(1080, 1350)) for _ in range(2)]

                    seg_outputs["variants"].append({
                        "headline": headline,
                        "body": body,
                        "cta": cta,
                        "channel": channel,
                        "personas": seg_personas,
                    })

                    cols = st.columns(2)
                    cols[0].image(posters[0], caption=f"Variant {v+1} Visual A", use_column_width=True)
                    cols[1].image(posters[1], caption=f"Variant {v+1} Visual B", use_column_width=True)

                    # Mini-metrics (simulated)
                    lift = seeded_score(brand_desc, seg_title, goal, tone, v)
                    st.caption(f"Predicted Engagement Lift: **{lift}%**  •  Suggested: **{channel}**")

                st.session_state["segments"].append(seg_outputs)

            # Assemble artifacts for ZIP
            images_for_zip = []
            for s_i, seg in enumerate(st.session_state["segments"]):
                for v_i, variant in enumerate(seg["variants"]):
                    for ab in ["A", "B"]:
                        img = draw_poster(variant["headline"], variant["body"], variant["cta"], tone, size=(1080, 1350))
                        path = f"images/segment{s_i+1}_variant{v_i+1}_{ab}.png"
                        images_for_zip.append({"image": img, "path": path})

            copy_payload = {
                "brand": brand_desc,
                "goal": goal,
                "tone": tone,
                "segments": st.session_state["segments"],
                "meta": {
                    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                    "lite_mode": bool(lite_mode or not HAS_MODELS),
                }
            }
            st.session_state["artifacts"] = {"copy": copy_payload, "images": images_for_zip, "tag": now_tag}
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
                        f"  _Channel: {v['channel']}_"
                    )

            zip_bytes = package_zip(st.session_state["artifacts"])
            name = f"ritual_ads_poc_{st.session_state['artifacts']['tag']}.zip"
            st.download_button(
                "⬇️ Download ZIP (PNG + copy.json)",
                data=zip_bytes,
                file_name=name,
                mime="application/zip",
                use_container_width=True,
            )
            st.caption("All assets are generated locally. Modify tone/personas, re-run, and download again.")


if __name__ == "__main__":
    main()
