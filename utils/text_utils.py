from __future__ import annotations
from typing import List, Dict, Tuple
import re, math
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------- clustering ----------

def cluster_personas(personas: List[str], k: int) -> List[str]:
    """TF-IDF + KMeans. If fewer personas than k, returns unique personas."""
    uniq = [p for i,p in enumerate(personas) if p not in personas[:i]]
    if len(uniq) <= k: return uniq
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit(uniq)
    X = vec.transform(uniq)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    segments: List[str] = []
    for c in range(k):
        idxs = [i for i,l in enumerate(kmeans.labels_) if l==c]
        # choose the most "central" persona in the cluster as label
        center = kmeans.cluster_centers_[c]
        best_i = max(idxs, key=lambda i: (X[i] @ center).A.ravel()[0])
        segments.append(uniq[best_i])
    return segments

# ---------- LLM copy (optional) ----------

_LLM = None
def _maybe_llm():
    global _LLM
    if _LLM is not None: return _LLM
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        _LLM = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
        return _LLM
    except Exception:
        _LLM = False
        return False

def _llm_copy(brand_desc: str, goal: str, voice: str, persona: str) -> Dict:
    llm = _maybe_llm()
    if not llm: raise RuntimeError("LLM unavailable")
    prompt = (
        f"You are a senior ad creative.\n"
        f"Brand: {brand_desc}\n"
        f"Goal: {goal}\nVoice: {voice}\nAudience: {persona}\n\n"
        "Write a headline (max 8 words), a body (<=40 words), and a clear CTA (2–3 words). "
        "Return as JSON with keys headline, body, cta."
    )
    out = llm(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"]
    # light JSON recovery
    m = re.search(r'\{.*\}', out, flags=re.S)
    if not m: raise RuntimeError("No JSON in LLM output")
    try:
        import json
        j = json.loads(m.group(0))
        return {"headline": j.get("headline",""), "body": j.get("body",""), "cta": j.get("cta","")}
    except Exception:
        raise RuntimeError("LLM JSON parse failed")

# ---------- template fallback ----------

def _template_copy(brand_desc: str, goal: str, voice: str, persona: str) -> Dict:
    voice_tag = f" ({voice})" if voice else ""
    if goal == "Awareness":
        h = f"Join In — {persona.title()}"
        b = f"For {persona}, {brand_desc.split('—')[0].strip().rstrip('.')}: creative rituals scale your story{voice_tag}."
        c = "Learn More"
    elif goal == "Signups/Leads":
        h = f"Unlock {persona.title()}"
        b = f"{brand_desc.split('—')[0].strip().rstrip('.')} — fast assets for {persona}, clear next steps{voice_tag}."
        c = "Start Free"
    elif goal == "Sales/Conversions":
        h = f"Switch to Ritual Ads"
        b = f"{persona.title()} get signal-driven creative that converts — built by AI operators{voice_tag}."
        c = "Get a Demo"
    else:
        h = f"Turn {persona.title()} Rituals into Results"
        b = f"Blend human-centered creative with signal-driven precision — for {persona}{voice_tag}."
        c = "Join In"
    return {"headline": h, "body": b, "cta": c}

def generate_variants_for_segment(brand_desc: str, goal: str, voice: str, persona_seg: str, n_variants: int) -> List[Dict]:
    out = []
    for _ in range(n_variants):
        try:
            out.append(_llm_copy(brand_desc, goal, voice, persona_seg))
        except Exception:
            out.append(_template_copy(brand_desc, goal, voice, persona_seg))
    return out

# ---------- scoring / guardrails ----------

def _readability_flesch_en(text: str) -> float:
    # quick Flesch Reading Ease approximation
    words = re.findall(r"[A-Za-z]+", text)
    sents = max(1, len(re.findall(r"[.!?]", text)))
    syll = sum(_syllables_en(w) for w in words)
    W = max(1, len(words))
    return 206.835 - 1.015*(W/sents) - 84.6*(syll/W)

def _syllables_en(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0; prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel: count += 1
        prev_vowel = is_v
    if word.endswith("e") and count>1: count -= 1
    return max(1, count)

BANNED = {"guarantee","best-ever","no-risk","get rich","risk-free"}

def score_copy_package(copy: Dict, brand_desc: str, goal: str) -> Dict:
    headline, body, cta = copy["headline"], copy["body"], copy["cta"]
    # length obeys constraints
    h_len = len(headline.split())
    h_score = max(0, 100 - max(0, h_len-8)*12)  # penalize >8 words
    b_len = len(body.split())
    b_score = max(0, 100 - max(0, b_len-40)*2)
    # readability
    r = _readability_flesch_en(body)
    r_score = min(100, max(0, (r-30)*3))  # normalize approx
    # brand keyword hit
    brand_tokens = set(re.findall(r"[a-zA-Z]+", brand_desc.lower()))
    body_tokens = set(re.findall(r"[a-zA-Z]+", (headline+" "+body).lower()))
    overlap = len(brand_tokens & body_tokens)
    k_score = min(100, overlap * 5)
    # guardrails / compliance
    bad_hits = [w for w in BANNED if w in (headline+" "+body).lower()]
    g_penalty = 25*len(bad_hits)
    total = max(0, round((0.30*h_score + 0.35*b_score + 0.20*r_score + 0.15*k_score) - g_penalty))
    return {
        "headline_len": h_len, "body_len": b_len,
        "readability": round(r,1),
        "brand_overlap": overlap,
        "banned_hits": bad_hits,
        "total": total
    }
