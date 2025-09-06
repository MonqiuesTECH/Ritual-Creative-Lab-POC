from __future__ import annotations
from typing import List, Dict

def generate_segments(personas: List[str], k: int) -> List[str]:
    """Pick top-k distinct personas (simple heuristic for POC)."""
    unique = []
    for p in personas:
        if p not in unique:
            unique.append(p)
    return unique[:max(1, k)]

def variant_copy_for_segment(brand_desc: str, goal: str, voice: str, persona: str, variant: int) -> Dict:
    # Tiny, goal-aware templates (free + deterministic)
    voice_tag = f" ({voice})" if voice else ""
    if goal == "Awareness":
        headline = f"Join In — {persona.title()}"
        body = f"For {persona}, {brand_desc.split('—')[0].strip().rstrip('.')}: creative built as repeatable rituals that scale your story{voice_tag}."
        cta = "Learn More"
        channel = "Instagram Carousel / LinkedIn Post"
    elif goal == "Signups/Leads":
        headline = f"Unlock {persona.title()}"
        body = f"For {persona}, ZARI turns repeatable work into reliable rituals — fast assets, on-brand copy, and clear next steps{voice_tag}."
        cta = "Start Free"
        channel = "LinkedIn Lead Gen / Landing Page"
    elif goal == "Sales/Conversions":
        headline = f"Switch to Ritual Ads"
        body = f"{persona.title()} get signal-driven creative that converts — built by AI operators to stay on-brand and on-time{voice_tag}."
        cta = "Get a Demo"
        channel = "Landing Page / Email"
    else:  # Engagement
        headline = f"Turn {persona.title()} Rituals into Results"
        body = f"Blend human-centered creative with signal-driven precision — ZARI builds reliable rituals for {persona}{voice_tag}."
        cta = "Join In"
        channel = "Instagram Reel / LinkedIn"
    return {
        "headline": headline,
        "body": body,
        "cta": cta,
        "channel": channel,
    }
