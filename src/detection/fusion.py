"""Combine model probabilities with rule weights. No random numbers are used."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    LABELS,
    MODEL_BLEND,
    MODEL_CONFIDENCE_FLOOR,
    MODEL_RISK_FRAUD,
    MODEL_RISK_SAFE,
    MODEL_RISK_SUSPICIOUS,
    RISK_FRAUD,
    RISK_SUSPICIOUS,
    RULE_BLEND,
    RULE_RISK_CAP,
)
from src.detection.indicators import Indicator

POLICY_SUMMARY = (
    "The text model is a TF-IDF logistic regression trained on this project's synthetic corpus. "
    f"Its confidence is the highest class probability. A model label is kept when confidence is at least "
    f"{MODEL_CONFIDENCE_FLOOR:.0%}. Below that, the blended risk score chooses the label: "
    f"SAFE under {RISK_SUSPICIOUS:.0f}, SUSPICIOUS under {RISK_FRAUD:.0f}, otherwise FRAUD. "
    "Any critical indicator forces FRAUD. "
    f"Blended risk = {MODEL_BLEND:.2f} × model risk + {RULE_BLEND:.2f} × rule risk, clipped to 0–100. "
    f"Model risk = {MODEL_RISK_FRAUD:.0f}×P(FRAUD) + {MODEL_RISK_SUSPICIOUS:.0f}×P(SUSPICIOUS) + "
    f"{MODEL_RISK_SAFE:.0f}×P(SAFE). Rule risk is the sum of indicator weights, capped at {RULE_RISK_CAP:.0f}. "
    "URL-only scans skip the text model and publish the structural score instead."
)


@dataclass
class Fusion:
    classification: str
    risk_score: float
    model_label: str | None
    model_confidence: float | None
    class_probabilities: dict[str, float] | None
    model_risk: float | None
    rule_risk: float
    policy_notes: list[str] = field(default_factory=list)
    explanation: str = ""


def model_risk(probabilities: dict[str, float]) -> float:
    return (
        probabilities.get("FRAUD", 0.0) * MODEL_RISK_FRAUD
        + probabilities.get("SUSPICIOUS", 0.0) * MODEL_RISK_SUSPICIOUS
        + probabilities.get("SAFE", 0.0) * MODEL_RISK_SAFE
    )


def rule_risk(indicators: list[Indicator]) -> float:
    return min(RULE_RISK_CAP, float(sum(item.weight for item in indicators)))


def _band(score: float) -> str:
    if score >= RISK_FRAUD:
        return "FRAUD"
    if score >= RISK_SUSPICIOUS:
        return "SUSPICIOUS"
    return "SAFE"


def fuse(
    indicators: list[Indicator],
    *,
    model_label: str | None = None,
    probabilities: dict[str, float] | None = None,
) -> Fusion:
    rules = rule_risk(indicators)
    notes: list[str] = []
    if probabilities is None or model_label is None:
        score = round(min(100.0, rules), 1)
        label = "FRAUD" if any(item.severity == "critical" for item in indicators) else _band(score)
        if any(item.severity == "critical" for item in indicators) and label != _band(score):
            notes.append(
                f"A critical indicator set the classification to {label}. The structural score is {score:.1f}."
            )
        explanation = _explain(
            classification=label,
            risk_score=score,
            model_label=None,
            model_confidence=None,
            probabilities=None,
            model_component=None,
            rules=rules,
            indicators=indicators,
            notes=notes,
        )
        return Fusion(
            classification=label,
            risk_score=score,
            model_label=None,
            model_confidence=None,
            class_probabilities=None,
            model_risk=None,
            rule_risk=round(rules, 1),
            policy_notes=notes,
            explanation=explanation,
        )

    missing = [label for label in LABELS if label not in probabilities]
    if missing:
        raise ValueError(f"Model probabilities are missing labels: {', '.join(missing)}")
    component = model_risk(probabilities)
    blended = min(100.0, MODEL_BLEND * component + RULE_BLEND * rules)
    score = round(blended, 1)
    confidence = float(probabilities[model_label])
    if confidence >= MODEL_CONFIDENCE_FLOOR:
        label = model_label
        notes.append(
            f"Model confidence {confidence:.1%} is at least {MODEL_CONFIDENCE_FLOOR:.0%}, so the model label was the starting point."
        )
    else:
        label = _band(score)
        notes.append(
            f"Model confidence {confidence:.1%} is below {MODEL_CONFIDENCE_FLOOR:.0%}, so the blended score {score:.1f} selected {label}."
        )
    critical = [item.name for item in indicators if item.severity == "critical"]
    if critical and label != "FRAUD":
        label = "FRAUD"
        notes.append(
            "Critical indicators (" + ", ".join(critical) + ") forced the classification to FRAUD."
        )
    explanation = _explain(
        classification=label,
        risk_score=score,
        model_label=model_label,
        model_confidence=confidence,
        probabilities=probabilities,
        model_component=component,
        rules=rules,
        indicators=indicators,
        notes=notes,
    )
    return Fusion(
        classification=label,
        risk_score=score,
        model_label=model_label,
        model_confidence=confidence,
        class_probabilities={label: round(float(probabilities[label]), 6) for label in LABELS},
        model_risk=round(component, 2),
        rule_risk=round(rules, 1),
        policy_notes=notes,
        explanation=explanation,
    )


def _explain(
    *,
    classification: str,
    risk_score: float,
    model_label: str | None,
    model_confidence: float | None,
    probabilities: dict[str, float] | None,
    model_component: float | None,
    rules: float,
    indicators: list[Indicator],
    notes: list[str],
) -> str:
    sentences = [
        f"Final classification: {classification}.",
        f"Blended risk score: {risk_score:.1f} out of 100.",
    ]
    if model_label is None or probabilities is None or model_component is None or model_confidence is None:
        sentences.append(
            "The text classifier was not used. This score is the capped sum of structural or header rules."
        )
    else:
        pretty = ", ".join(f"{label} {probabilities[label]:.1%}" for label in LABELS)
        sentences.append(
            f"The text model predicted {model_label} with confidence {model_confidence:.1%}. "
            f"Class probabilities: {pretty}."
        )
        sentences.append(
            f"Model risk component = {MODEL_RISK_FRAUD:.0f}×{probabilities['FRAUD']:.3f} + "
            f"{MODEL_RISK_SUSPICIOUS:.0f}×{probabilities['SUSPICIOUS']:.3f} + "
            f"{MODEL_RISK_SAFE:.0f}×{probabilities['SAFE']:.3f} = {model_component:.2f}. "
            f"Rule risk component = {rules:.1f}. "
            f"Blended = {MODEL_BLEND:.2f}×{model_component:.2f} + {RULE_BLEND:.2f}×{rules:.1f} = {risk_score:.1f}."
        )
    if indicators:
        rendered = "; ".join(f"{item.name} ({item.severity}, {item.weight:g} points)" for item in indicators)
        sentences.append(f"Indicators: {rendered}.")
    else:
        sentences.append("No rule indicators fired.")
    sentences.extend(notes)
    sentences.append(
        "Trained on synthetic messages written for this demo. A score is a decision-support signal, not proof of fraud or safety."
    )
    return " ".join(sentences)
