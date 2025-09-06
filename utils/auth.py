import streamlit as st

def require_password(password: str | None):
    """Gate the app if APP_PASSWORD is set."""
    if not password:
        return
    if "authed" not in st.session_state:
        st.session_state["authed"] = False
    if not st.session_state["authed"]:
        st.title("🔐 Enter Access Password")
        pw = st.text_input("Password", type="password")
        if st.button("Enter") and pw == password:
            st.session_state["authed"] = True
            st.rerun()
        st.stop()
