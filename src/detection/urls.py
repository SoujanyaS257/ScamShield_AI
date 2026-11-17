"""Structural URL scoring.

The app does not fetch the URL. Fetching user-supplied addresses would create
server-side request forgery risk and would leak the analyst's IP to the host.
Only the string is scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.detection.indicators import Indicator

SUSPICIOUS_TLDS = {
    "zip",
    "mov",
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "work",
    "click",
    "country",
    "support",
    "loan",
    "top",
    "xyz",
    "cam",
    "rest",
    "monster",
    "quest",
    "icu",
}

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "cutt.ly",
    "is.gd",
    "rb.gy",
    "tiny.cc",
    "shorturl.at",
    "rebrand.ly",
}

# Brand token -> official registrable domains. Used only as an allow-list.
BRANDS: dict[str, set[str]] = {
    "paypal": {"paypal.com"},
    "apple": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "outlook.com", "office365.com"},
    "amazon": {"amazon.com", "amazon.in", "amazon.co.uk"},
    "google": {"google.com", "gmail.com", "youtube.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "irs": {"irs.gov"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "dhl": {"dhl.com"},
    "hdfc": {"hdfcbank.com", "hdfc.bank.in"},
    "icici": {"icicibank.com", "icici.bank.in"},
    "sbi": {"onlinesbi.sbi", "sbi.co.in", "sbi.bank.in"},
}

MULTI_SUFFIXES = {
    "co.uk",
    "com.au",
    "co.in",
    "com.br",
    "co.za",
    "com.mx",
    "bank.in",
    "co.jp",
}

PATH_KEYWORDS = {
    "login": 12,
    "signin": 12,
    "sign-in": 12,
    "verify": 12,
    "password": 14,
    "passwd": 14,
    "wallet": 10,
    "invoice": 8,
    "banking": 10,
    "kyc": 12,
    "unlock": 10,
    "credential": 12,
}

CONFUSABLES = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})

POINTS_NO_HTTPS = 14
POINTS_IP = 36
POINTS_USERINFO = 36
POINTS_SHORTENER = 35
POINTS_PUNYCODE = 18
POINTS_SUSPICIOUS_TLD = 12
POINTS_HYPHENS = 10
POINTS_LONG = 8
POINTS_SUBDOMAINS = 12
POINTS_ODD_PORT = 8
POINTS_LOOKALIKE = 40
POINTS_BRAND_MISUSE = 36
POINTS_DANGEROUS_SCHEME = 80
KEYWORD_CAP = 24


@dataclass
class UrlAssessment:
    normalized_url: str
    risk_score: float
    classification: str
    indicators: list[Indicator] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "normalized_url": self.normalized_url,
            "risk_score": self.risk_score,
            "classification": self.classification,
            "indicators": [item.as_dict() for item in self.indicators],
            "findings": list(self.findings),
        }


def registrable_domain(host: str) -> str:
    parts = [part for part in host.lower().strip(".").split(".") if part]
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIXES:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return ".".join(parts)


def is_official_host(host: str) -> bool:
    domain = registrable_domain(host)
    return any(domain in domains for domains in BRANDS.values())


def _encode_host(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("The domain contains characters that are not valid in a hostname.") from exc


def _is_ip(host: str) -> bool:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        octets = [int(part) for part in host.split(".")]
        return all(0 <= octet <= 255 for octet in octets)
    return ":" in host and re.fullmatch(r"[0-9a-f:]+", host) is not None


def _brand_indicators(host: str) -> list[Indicator]:
    if is_official_host(host):
        return []
    indicators: list[Indicator] = []
    labels = [label for label in host.split(".") if label and label != "www"]
    domain = registrable_domain(host)
    for brand, official in BRANDS.items():
        if domain in official:
            continue
        boundary = re.search(rf"(^|[^a-z]){re.escape(brand)}([^a-z]|$)", host)
        lookalike = False
        prefixed = False
        for label in labels:
            folded = label.translate(CONFUSABLES)
            if folded == brand and label != brand:
                lookalike = True
            if label.startswith(brand) and label != brand and len(label) <= len(brand) + 4:
                prefixed = True
        if lookalike:
            indicators.append(
                Indicator(
                    name="brand_lookalike",
                    severity="critical",
                    weight=POINTS_LOOKALIKE,
                    detail=f"A hostname label imitates '{brand}' by swapping characters (host {host}).",
                )
            )
        elif boundary or prefixed:
            indicators.append(
                Indicator(
                    name="brand_impersonation",
                    severity="critical",
                    weight=POINTS_BRAND_MISUSE,
                    detail=(
                        f"The host mentions '{brand}' but the registrable domain is {domain}, "
                        f"not an official {brand} domain ({', '.join(sorted(official))})."
                    ),
                )
            )
    return indicators


def score_url(url: str) -> UrlAssessment:
    """Score one URL string. Raises ValueError when it cannot be parsed."""
    raw = url.strip()
    supplied_scheme = "://" in raw
    candidate = raw if supplied_scheme else f"http://{raw}"
    parsed = urlparse(candidate)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"javascript", "data", "vbscript", "file"}:
        indicator = Indicator(
            name="dangerous_scheme",
            severity="critical",
            weight=POINTS_DANGEROUS_SCHEME,
            detail=f"The URL uses the '{scheme}' scheme, which is not a normal web link.",
        )
        return UrlAssessment(
            normalized_url=candidate,
            risk_score=100.0,
            classification="FRAUD",
            indicators=[indicator],
            findings=["dangerous scheme"],
        )
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme '{scheme or 'none'}'. Use an http or https URL.")
    if not parsed.hostname:
        raise ValueError("The URL does not contain a hostname.")

    host = _encode_host(parsed.hostname)
    indicators: list[Indicator] = []
    score = 0.0

    def add(name: str, severity: str, points: float, detail: str) -> None:
        nonlocal score
        if points <= 0:
            return
        score += points
        indicators.append(Indicator(name=name, severity=severity, weight=points, detail=detail))

    if parsed.username or parsed.password:
        add(
            "url_userinfo",
            "critical",
            POINTS_USERINFO,
            "The URL embeds a username or password before the host. Browsers can hide the real destination that way.",
        )
    if scheme != "https":
        reason = "No scheme was provided, so the address was scored as plain http." if not supplied_scheme else "The link does not use HTTPS."
        add("no_https", "warning", POINTS_NO_HTTPS, reason)
    if _is_ip(host):
        severity = "critical" if re.search(r"login|verify|account|password|wallet|kyc", parsed.path or "", re.I) else "warning"
        add("ip_host", severity, POINTS_IP, f"The host is an IP address ({host}) rather than a domain name.")
    if registrable_domain(host) in SHORTENERS or host in SHORTENERS:
        add(
            "shortened_url",
            "warning",
            POINTS_SHORTENER,
            "This is a link shortener. The destination is hidden, and ScamShield does not open it.",
        )
    if host.startswith("xn--") or ".xn--" in host:
        add("punycode_host", "warning", POINTS_PUNYCODE, "The host uses punycode, which can disguise look-alike international characters.")
    tld = host.rsplit(".", 1)[-1]
    if tld in SUSPICIOUS_TLDS:
        add("suspicious_tld", "warning", POINTS_SUSPICIOUS_TLD, f"The top-level domain '.{tld}' is frequently abused. Legitimate sites use it too.")
    if host.count("-") >= 3:
        add("hyphenated_host", "info", POINTS_HYPHENS, "The hostname contains several hyphens, a common pattern in disposable phishing domains.")
    if len(candidate) > 90:
        add("long_url", "info", POINTS_LONG, "The URL is unusually long.")
    if len([part for part in host.split(".") if part]) >= 5:
        add("many_subdomains", "warning", POINTS_SUBDOMAINS, "The hostname has a deep stack of subdomains.")
    if parsed.port and parsed.port not in {80, 443}:
        add("unusual_port", "warning", POINTS_ODD_PORT, f"The URL uses network port {parsed.port}.")

    brand_hits = _brand_indicators(host)
    indicators.extend(brand_hits)
    score += sum(item.weight for item in brand_hits)

    if not is_official_host(host):
        haystack = f"{parsed.path or ''} {parsed.query or ''}".lower()
        keyword_points = 0
        matched: list[str] = []
        for keyword, points in PATH_KEYWORDS.items():
            if keyword in haystack:
                keyword_points += points
                matched.append(keyword)
        keyword_points = min(KEYWORD_CAP, keyword_points)
        if keyword_points:
            add(
                "sensitive_path",
                "warning",
                keyword_points,
                "The path or query asks for a sensitive action (" + ", ".join(matched) + ") on a non-official host.",
            )

    score = round(min(100.0, score), 1)
    critical = any(item.severity == "critical" for item in indicators)
    if critical or score >= 70:
        label = "FRAUD"
    elif score >= 35 or any(item.name == "shortened_url" for item in indicators):
        label = "SUSPICIOUS"
    else:
        label = "SAFE"
    return UrlAssessment(
        normalized_url=candidate,
        risk_score=score,
        classification=label,
        indicators=indicators,
        findings=[item.name for item in indicators],
    )
