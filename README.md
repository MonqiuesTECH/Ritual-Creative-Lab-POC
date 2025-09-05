# Ritual Creative Lab — POC (Powered by ZARI)

A free, Streamlit-based POC that generates persona-tailored ad campaigns:
- On-brand copy (headline, body, CTA) via a **local CPU model** (FLAN-T5-small) with a **graceful Lite Mode fallback**
- Persona clustering using **Sentence-Transformers (MiniLM)** + KMeans (CPU)
- Programmatic poster-style **visuals** (PNG) with Pillow — no paid image APIs
- One-click **ZIP export** of PNGs + `copy.json`
- “Enterprise polish”: demo personas, integration placeholders, simulated analytics, brand theme

> **No API keys. No paid services. Streamlit Cloud friendly.**

---

## Quickstart

### Local
```bash
git clone https://github.com/<your-org>/ritual-creative-lab-poc.git
cd ritual-creative-lab-poc
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
