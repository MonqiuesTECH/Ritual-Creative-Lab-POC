import streamlit as st

def brand_form():
    st.caption("Brand description (who/why/ritual)")
    desc = st.text_area("", height=80, placeholder="What your brand does, why it matters, and the ritual/context of use.")
    st.caption("Primary goal")
    goal = st.selectbox("", ["Engagement", "CTR", "Leads", "Awareness"], index=0)
    st.caption("Voice / Tone")
    voice = st.text_input("", value="Bold, modern, human-centered")
    return {"description": desc, "primary_goal": goal, "voice_tone": voice}

def persona_form():
    st.caption("Personas (one per line)")
    raw = st.text_area("", height=80, placeholder="Fitness-curious millennials\nCTOs at seed-stage SaaS")
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    return lines

def footer_brand_bar():
    return "Ritual Ads • Powered by ZARI"
