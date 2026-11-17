"""ScamShield AI dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.pages import (
    render_about,
    render_analytics,
    render_email,
    render_history,
    render_home,
    render_image,
    render_qr,
    render_sms,
    render_transcript,
    render_url,
    render_voice,
)
from frontend.theme import inject_css
from src.database import open_store
from src.ml.classifier import get_classifier

PAGES = [
    ("Home", "home", render_home),
    ("SMS & Messages", "sms", render_sms),
    ("Email", "email", render_email),
    ("Website & URL", "url", render_url),
    ("QR Code", "qr", render_qr),
    ("Image", "image", render_image),
    ("Voice", "voice", render_voice),
    ("Phone Transcript", "transcript", render_transcript),
    ("History", "history", render_history),
    ("Analytics", "analytics", render_analytics),
    ("About", "about", render_about),
]


@st.cache_resource
def _store():
    return open_store()


def main() -> None:
    st.set_page_config(
        page_title="ScamShield AI",
        page_icon="🛡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    with st.spinner("Loading the text model…"):
        get_classifier()
    store = _store()
    page = _navigation(store)
    renderer = dict((key, func) for _, key, func in PAGES)[page]
    renderer(store)


def _navigation(store) -> str:
    st.sidebar.markdown("### ScamShield AI")
    st.sidebar.caption("Defensive review demo. A label is not proof.")
    requested = st.query_params.get("page", "home")
    known = {key for _, key, _ in PAGES}
    current = requested if requested in known else "home"
    for label, key, _renderer in PAGES:
        if st.sidebar.button(
            label,
            key=f"nav-{key}",
            type="primary" if current == key else "secondary",
            width="stretch",
        ):
            st.query_params["page"] = key
            st.rerun()
    if store.warning:
        st.sidebar.warning(store.warning)
    else:
        st.sidebar.success(f"Storage: {store.mode}")
    return current


if __name__ == "__main__":
    main()
