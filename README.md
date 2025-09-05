# Ritual Creative Lab — POC (Powered by ZARI)

A **free**, **Streamlit-based** proof-of-concept that generates persona-tailored ad campaigns:
- On-brand copy (headline, body, CTA) with a **local CPU model** (FLAN-T5-small)
- Persona clustering using **Sentence-Transformers (MiniLM)** on CPU
- Programmatic **ad visuals** (PNG posters) using **PIL** — no paid image APIs
- One-click **ZIP export** of creatives + copy JSON

> No API keys. No paid services. Runs on Streamlit Cloud free tier.

---

## ✨ Why this POC
Ritual Ads’ thesis blends **ritual + human-centered creativity** with **cutting-edge AI**.  
This POC demonstrates an **enterprise-ready path** using free/open components that can later plug into Google/Meta Ads, CRMs, and performance feedback loops.

---

## 🧱 Architecture (POC)
- **UI**: Streamlit single-file app
- **Copy Gen**: `google/flan-t5-small` via `transformers`
- **Embeddings/Clustering**: `sentence-transformers/all-MiniLM-L6-v2` + KMeans
- **Visuals**: PIL gradient/typographic posters (brandable palettes)
- **Packaging**: ZIP (PNG + `copy.json`)

---

## 🚀 Quickstart

### 1) Local
```bash
git clone https://github.com/<your-org>/ritual-creative-lab-poc.git
cd ritual-creative-lab-poc
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
