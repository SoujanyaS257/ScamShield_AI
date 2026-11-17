"""Explainable indicator records. Weights are fixed policy values, not probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEVERITIES = ("info", "warning", "critical")


@dataclass(frozen=True)
class Indicator:
    name: str
    severity: str
    weight: float
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {self.severity}")
        if self.weight < 0:
            raise ValueError("Indicator weight cannot be negative.")

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["weight"] = round(float(self.weight), 1)
        return payload


def dedupe(indicators: list[Indicator]) -> list[Indicator]:
    best: dict[str, Indicator] = {}
    for indicator in indicators:
        current = best.get(indicator.name)
        if current is None or indicator.weight > current.weight:
            best[indicator.name] = indicator
    return sorted(best.values(), key=lambda item: (-item.weight, item.name))
