"""Dark cybersecurity theme."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
[data-testid="stSidebar"] {
  background: #0b1220;
  border-right: 1px solid #1e293b;
}
.block-container { padding-top: 1.4rem; max-width: 1180px; }
.shield-badge {
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.06em;
  font-size: 0.82rem;
}
.shield-badge.safe { background: #064e3b; color: #6ee7b7; }
.shield-badge.suspicious { background: #78350f; color: #fcd34d; }
.shield-badge.fraud { background: #881337; color: #fecdd3; }
.hero-kicker {
  color: #67e8f9;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}
div[data-testid="stMetric"] {
  background: #0f172a;
  border: 1px solid #1e293b;
  padding: 0.7rem 0.85rem;
  border-radius: 0.8rem;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
