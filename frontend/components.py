"""Shared Streamlit widgets."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.detection.engine import DetectionResult

LABEL_COLORS = {"SAFE": "#34d399", "SUSPICIOUS": "#fbbf24", "FRAUD": "#fb7185"}


def badge(label: str) -> None:
    st.markdown(
        f"<span class='shield-badge {label.lower()}'>{label}</span>",
        unsafe_allow_html=True,
    )


def show_result(result: DetectionResult) -> None:
    badge(result.classification)
    st.caption("Decision-support signal only. This does not prove the content is safe or fraudulent.")
    first, second, third, fourth = st.columns(4)
    first.metric("Blended risk", f"{result.risk_score:.1f}")
    second.metric(
        "Model confidence",
        "—" if result.model_confidence is None else f"{result.model_confidence:.1%}",
    )
    third.metric("Model risk", "—" if result.model_risk is None else f"{result.model_risk:.1f}")
    fourth.metric("Rule risk", f"{result.rule_risk:.1f}")
    if result.model_label:
        st.write(
            f"Text model label: **{result.model_label}**. "
            "Confidence is the highest class probability, not the blended risk score."
        )
    else:
        st.write("No text-model probability was produced for this scan.")
    for note in result.policy_notes:
        st.info(note)
    with st.container(border=True):
        st.markdown("**Why this result**")
        st.write(result.explanation)
    if result.indicators:
        st.markdown("**Indicators**")
        frame = pd.DataFrame([item.as_dict() for item in result.indicators])
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.caption("No rule indicators fired.")
    if result.class_probabilities:
        chart = pd.DataFrame(
            {
                "label": list(result.class_probabilities.keys()),
                "probability": list(result.class_probabilities.values()),
            }
        )
        figure = px.bar(
            chart,
            x="label",
            y="probability",
            color="label",
            color_discrete_map=LABEL_COLORS,
            range_y=[0, 1],
            title="Text model class probabilities",
        )
        _style(figure)
        st.plotly_chart(figure, width="stretch")
    if result.feature_contributions:
        with st.expander("Features that moved the linear model"):
            st.dataframe(pd.DataFrame(result.feature_contributions), width="stretch", hide_index=True)
            st.caption(
                "Each value is a TF-IDF weight times the coefficient for the model's chosen class. "
                "The intercept is excluded, so the numbers do not add up to the log-odds."
            )
    if result.normalized_text:
        with st.expander("Text the model actually scored"):
            st.code(result.normalized_text)
    if result.input_type in {"qr", "image", "voice"} and result.extracted_text:
        st.markdown("**Extracted content**")
        st.text_area(
            "Extracted content",
            result.extracted_text,
            height=140,
            disabled=True,
            label_visibility="collapsed",
        )


def _style(figure):
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(l=8, r=8, t=48, b=8),
        height=340,
        showlegend=False,
    )
    return figure
