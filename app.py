import io
import json
import zipfile
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

APP_TITLE = "Ritual Creative Lab — POC (Powered by ZARI)"
BRAND_BAR = "Ritual Ads • Powered by ZARI"

@st.cache_resource(show_spinner=False)
def load_models():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return gen, embedder

def gen_text(gen, prompt: str, max_tokens: int = 128) -> str:
    out = gen(
        prompt,
        max_length=max_tokens,
        do_sample=True,
        top_p=0.92,
        temperature=0.8,
        num_return_sequences=1
    )
    return out[0]["generated_text"].strip()

def safe_font(size=48):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except:
        return ImageFont.load_default()

def make_gradient(w, h, start_rgb, end_rgb):
    base = Image.new("RGB", (w, h), start_rgb)
    top = Image.new("RGB", (w, h), end_rgb)
    mask = Image.linear_gradient("L").resize((w, h))
    return Image.composite(top, base, mask)

def contrast_color(rgb):
    r, g, b = rgb
    yiq = (r*299 + g*587 + b*114) / 1000
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

    text_color = contrast_color(((start[0]+end[0])//2, (start[1]+end[1])//2, (start[2]+end[2])//2))

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
    footer = BRAND_BAR
    footer_w = draw.textlength(footer, font=f_brand)
    draw.text((w - pad - footer_w, h - pad - f_brand.size - 4), footer, font=f_brand, fill=text_color)

    return bg

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

def parse_copy(text: str):
    lines = text.splitlines()
    get = lambda key: next((l.split(":", 1)[1].strip() for l in lines if l.strip().lower().startswith(f"{key}:")), "")
    h = get("headline") or text[:70]
    b = get("body") or text
    c = get("cta") or "Learn More"
    return h, b, c

def cluster_personas(embedder, personas: List[str], k: int):
    if len(personas) == 0:
        return [], []
    k = max(1, min(k, len(personas)))
    X = embedder.encode(personas)
    if len(personas) == 1:
        return [0], ["Segment 1: Solo"]
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lab = km.fit_predict(X)
    summaries = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(X[idx] - centroid, axis=1)
        summaries.append(f"Segment {c+1}: like '{personas[idx[np.argmin(dists)]]}'")
    return lab, summaries

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

def ui_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
    st.markdown(
        f"""
        <div style="padding:10px 16px;border-radius:12px;background:#0f172a;color:#e2e8f0;display:flex;justify-content:space-between;align-items:center;">
            <div style="font-size:18px;font-weight:700;">{APP_TITLE}</div>
            <div style="opacity:0.85;">Free POC • Local models • No API keys</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    ui_header()
    gen, embedder = load_models()

    with st.sidebar:
        st.subheader("Brand & Campaign")
        brand_desc = st.text_area(
            "Brand description (who/why/ritual):",
            height=120,
            placeholder="We blend timeless ritual with human-centered AI to move hearts and drive results."
        )
        goal = st.selectbox("Primary goal", ["Awareness", "Engagement", "Signups/Leads", "Sales/Conversions"])
        tone = st.text_input("Voice / Tone", value="Bold, modern, human-centered")

        st.markdown("---")
        st.subheader("Personas (one per line)")
        personas_text = st.text_area("Examples: 'Fitness-curious millennials', 'CTOs at seed-stage SaaS', etc.", height=140)
        k_clusters = st.slider("Number of persona segments", min_value=1, max_value=6, value=2)

        st.markdown("---")
        n_variants = st.slider("Variants per segment", 1, 3, 2)
        st.caption("Branding: Ritual Ads • Powered by ZARI")

    personas = [p.strip() for p in personas_text.splitlines() if p.strip()]
    tabs = st.tabs(["Plan", "Generate", "Review & Download"])

    with tabs[0]:
        st.markdown("### Creative Plan")
        st.write(
            "- Cluster personas, generate copy per segment, auto-create visual posters.\n"
            "- Everything runs **locally** with open-source models on CPU.\n"
            "- Outputs: **headline, body, CTA**, and **2 PNG visuals** per variant.\n"
            "- Download a **ZIP** (PNG + copy.json) from the final tab."
        )
        if brand_desc:
            st.info(f"Brand Summary: {brand_desc}")
        if personas:
            st.success(f"{len(personas)} persona(s) provided.")

    with tabs[1]:
        st.markdown("### Generate Ad Packages")
        if st.button("Run Generator", type="primary", use_container_width=True):
            if not brand_desc or not personas:
                st.error("Please provide a brand description and at least one persona in the sidebar.")
                st.stop()

            labels, summaries = cluster_personas(embedder, personas, k_clusters)
            st.session_state["segments"] = []
            now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            for seg_idx, seg_title in enumerate(summaries):
                seg_personas = [p for p, lab in zip(personas, labels) if lab == seg_idx]
                st.write(f"#### {seg_title}")
                st.caption(f"Personas in this segment: {', '.join(seg_personas)}")

                seg_outputs = {"segment": seg_title, "variants": []}
                for v in range(n_variants):
                    prompt = make_copy_prompt(brand_desc, goal, tone, f"{seg_title}. People like: {', '.join(seg_personas[:3])}.")
                    raw = gen_text(gen, prompt, max_tokens=128)
                    headline, body, cta = parse_copy(raw)

                    # two visual variants
                    posters = [draw_poster(headline, body, cta, tone, size=(1080, 1350)) for _ in range(2)]
                    seg_outputs["variants"].append({"headline": headline, "body": body, "cta": cta})

                    cols = st.columns(2)
                    cols[0].image(posters[0], caption=f"Variant {v+1} Visual A", use_column_width=True)
                    cols[1].image(posters[1], caption=f"Variant {v+1} Visual B", use_column_width=True)

                st.session_state["segments"].append(seg_outputs)

            # Stash images to ZIP
            images_for_zip = []
            for s_i, seg in enumerate(st.session_state["segments"]):
                for v_i, variant in enumerate(seg["variants"]):
                    for ab in ["A", "B"]:
                        img = draw_poster(variant["headline"], variant["body"], variant["cta"], tone, size=(1080, 1350))
                        path = f"images/segment{s_i+1}_variant{v_i+1}_{ab}.png"
                        images_for_zip.append({"image": img, "path": path})

            copy_payload = {"brand": brand_desc, "goal": goal, "tone": tone, "segments": st.session_state["segments"]}
            st.session_state["artifacts"] = {"copy": copy_payload, "images": images_for_zip, "tag": now_tag}
            st.success("Generation complete. Review & download in the next tab.")

    with tabs[2]:
        st.markdown("### Review & Download")
        if "artifacts" not in st.session_state:
            st.info("Run the generator first.")
        else:
            # Preview copy
            for seg in st.session_state["artifacts"]["copy"]["segments"]:
                st.write(f"**{seg['segment']}**")
                for idx, v in enumerate(seg["variants"], start=1):
                    st.write(f"- **V{idx}** — **{v['headline']}**  \n  {v['body']}  \n  _CTA: {v['cta']}_")

            # ZIP download
            zip_bytes = package_zip(st.session_state["artifacts"])
            name = f"ritual_ads_poc_{st.session_state['artifacts']['tag']}.zip"
            st.download_button(
                "⬇️ Download ZIP (PNG + copy.json)",
                data=zip_bytes,
                file_name=name,
                mime="application/zip",
                use_container_width=True
            )
            st.caption("All assets are generated locally. Modify tone/personas, re-run, and download again.")

if __name__ == "__main__":
    main()
