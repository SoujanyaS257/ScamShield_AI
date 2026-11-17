"""High-precision text rules. A critical hit forces FRAUD in the decision policy."""

from __future__ import annotations

import re

from src.detection.indicators import Indicator, dedupe
from src.detection.urls import score_url

_URL = re.compile(r"(https?://[^\s<>'\"\]]+|www\.[^\s<>'\"\]]+)", re.IGNORECASE)

# Each rule: name, severity, weight, pattern, detail. Patterns are applied to raw text.
_RULES: tuple[tuple[str, str, float, re.Pattern[str], str], ...] = (
    (
        "otp_harvest",
        "critical",
        36,
        re.compile(
            r"\b(otp|one[-\s]?time (?:password|passcode|code)|verification code|security code|passcode)\b.{0,80}\b(share|send|tell|read|give|provide|forward|whatsapp)\b"
            r"|\b(share|send|tell|read|give|provide|forward)\b.{0,50}\b(otp|one[-\s]?time (?:password|passcode|code)|verification code|security code|passcode)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message asks someone to reveal a one-time code or passcode.",
    ),
    (
        "secret_request",
        "critical",
        36,
        re.compile(
            r"\b(upi\s*pin|atm\s*pin|card\s*pin|cvv|cvc|net\s*banking password|password|seed phrase|recovery phrase)\b.{0,60}\b(share|send|tell|give|enter|type|read|provide)\b"
            r"|\b(share|send|tell|give|provide|read)\b.{0,40}\b(upi\s*pin|atm\s*pin|cvv|cvc|password|seed phrase)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message asks for a password, PIN, card security code, or wallet recovery phrase.",
    ),
    (
        "gift_card_payment",
        "critical",
        34,
        re.compile(
            r"\b(gift\s*cards?|itunes|google\s*play\s*cards?|steam\s*cards?|apple\s*cards?)\b.{0,80}\b(buy|purchase|scratch|code|codes|read|send)\b"
            r"|\b(buy|purchase)\b.{0,40}\b(gift\s*cards?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message pressures the reader to buy gift cards and hand over the codes.",
    ),
    (
        "remote_access",
        "critical",
        32,
        re.compile(
            r"\b(anydesk|teamviewer|rustdesk|ultraviewer|remote\s+desktop|remote\s+access\s+software)\b",
            re.IGNORECASE,
        ),
        "The message mentions remote-access software, a common tech-support scam step.",
    ),
    (
        "payment_demand",
        "critical",
        30,
        re.compile(
            r"\b(wire|western\s+union|moneygram|bitcoin|crypto\s+wallet|usdt|bank\s+transfer|routing\s+number|processing\s+fee|release\s+fee|customs\s+fee)\b.{0,80}\b(pay|send|transfer|deposit|fee|urgent)\b"
            r"|\b(pay|send|transfer)\b.{0,40}\b(fee|bitcoin|usdt|wire|gift\s*card)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message demands a payment, transfer, or release fee.",
    ),
    (
        "prize_with_fee",
        "critical",
        30,
        re.compile(
            r"\b(you(?:'ve| have)? won|lottery|prize|inheritance|lucky\s+draw|grant approved|compensation)\b.{0,90}\b(fee|pay|transfer|claim|tax|release|charge)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message pairs a prize, grant, or inheritance with a fee to collect it.",
    ),
    (
        "job_fee",
        "critical",
        28,
        re.compile(
            r"\b(job|hiring|work\s+from\s+home|vacancy)\b.{0,70}\b(registration\s+fee|pay\s+to\s+apply|refundable\s+deposit|processing\s+fee|training\s+fee)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message asks for money in exchange for a job or hiring step.",
    ),
    (
        "authority_threat",
        "warning",
        16,
        re.compile(
            r"\b(arrest|warrant|crime\s+branch|income\s+tax|irs|customs|deportation|lawsuit|police\s+case|money\s+laundering)\b",
            re.IGNORECASE,
        ),
        "The message invokes a legal or government threat.",
    ),
    (
        "call_control",
        "warning",
        14,
        re.compile(r"\b(do not hang up|stay on the line|do not tell (?:anyone|your)|keep this confidential)\b", re.IGNORECASE),
        "The message tries to keep the person isolated or on the call.",
    ),
    (
        "kyc_pressure",
        "warning",
        18,
        re.compile(
            r"\bkyc\b.{0,60}\b(expire|expired|blocked|suspend|update|verify|link|click)\b|\b(update|verify)\b.{0,30}\bkyc\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message pressures the reader to update KYC, often through a link.",
    ),
    (
        "account_alarm",
        "warning",
        14,
        re.compile(
            r"\b(suspended|locked|disabled|unusual (?:activity|sign-?in)|compromised)\b.{0,50}\b(password|verify|login|click|restore|confirm your identity|immediately)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message combines an account alarm with a demand to verify or sign in.",
    ),
    (
        "urgency",
        "info",
        8,
        re.compile(
            r"\b(act now|immediately|within \d+ minutes|right now|final warning|last chance|expires? today)\b",
            re.IGNORECASE,
        ),
        "The wording pushes for immediate action.",
    ),
    (
        "forwarded_blast",
        "warning",
        10,
        re.compile(
            r"\b(forwarded many times|send this to (?:all|everyone)|share (?:this )?with (?:all )?your contacts|forward to \d+ (?:people|groups))\b",
            re.IGNORECASE,
        ),
        "The message asks to be forwarded widely.",
    ),
    (
        "relative_emergency",
        "warning",
        18,
        re.compile(
            r"\b(grandson|granddaughter|grandchild|nephew|niece)\b.{0,80}\b(jail|bail|accident|hospital|wire|stuck)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message claims a relative is in an emergency and needs money.",
    ),
    (
        "executive_impersonation",
        "warning",
        18,
        re.compile(
            r"\b(i am the ceo|this is the ceo|from the director|finance director)\b.{0,80}\b(gift card|wire|confidential|do not tell|urgent|payment)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "The message pretends to be an executive requesting a secret payment.",
    ),
)


def _negated(text: str, start: int) -> bool:
    """True when a denial sits in the same clause, immediately before the match.

    "Do not hang up. Read the OTP" must still match. "We will never ask for your
    password" must not.
    """
    window = text[max(0, start - 48) : start].lower()
    return re.search(r"\b(never|do not|don't|will not|won't)\b[^.!?\n]{0,32}$", window) is not None


def extract_urls(text: str) -> list[str]:
    found = []
    for match in _URL.findall(text or ""):
        cleaned = match.rstrip(".,);]>\"'")
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found


def _link_indicators(text: str) -> list[Indicator]:
    indicators: list[Indicator] = []
    for url in extract_urls(text):
        try:
            assessment = score_url(url)
        except ValueError:
            continue
        if assessment.classification == "SAFE" and assessment.risk_score < 20:
            continue
        severity = "critical" if assessment.classification == "FRAUD" else "warning"
        weight = min(40.0, assessment.risk_score * 0.55)
        if weight < 8:
            continue
        detail = f"{url} scored {assessment.risk_score:.0f}/100 from URL rules"
        if assessment.findings:
            detail += " (" + ", ".join(assessment.findings) + ")"
        indicators.append(
            Indicator(
                name="suspicious_link",
                severity=severity,
                weight=round(weight, 1),
                detail=detail,
            )
        )
    return indicators


def text_indicators(text: str, *, forwarded: bool = False) -> list[Indicator]:
    raw = text or ""
    found: list[Indicator] = []
    for name, severity, weight, pattern, detail in _RULES:
        match = pattern.search(raw)
        if not match or _negated(raw, match.start()):
            continue
        snippet = " ".join(match.group(0).split())[:140]
        found.append(
            Indicator(
                name=name,
                severity=severity,
                weight=weight,
                detail=f"{detail} Matched: “{snippet}”.",
            )
        )
    found.extend(_link_indicators(raw))
    if forwarded:
        found.append(
            Indicator(
                name="marked_forwarded",
                severity="info",
                weight=6,
                detail="The sender marked this WhatsApp message as forwarded. Forwarding alone is not proof of fraud.",
            )
        )
    return dedupe(found)


def email_header_indicators(
    sender: str,
    reply_to: str,
    attachment: str,
) -> list[Indicator]:
    from src.detection.urls import BRANDS, registrable_domain

    indicators: list[Indicator] = []
    display, address = _split_mailbox(sender)
    reply_display, reply_address = _split_mailbox(reply_to) if reply_to.strip() else ("", "")
    sender_domain = _domain(address)
    reply_domain = _domain(reply_address)

    if display and sender_domain:
        folded = display.lower()
        for brand, official in BRANDS.items():
            if re.search(rf"\b{re.escape(brand)}\b", folded) and registrable_domain(sender_domain) not in official:
                indicators.append(
                    Indicator(
                        name="display_name_impersonation",
                        severity="critical",
                        weight=32,
                        detail=(
                            f"The display name mentions {brand}, but the sender domain "
                            f"{registrable_domain(sender_domain)} is not an official {brand} domain."
                        ),
                    )
                )
                break
    if sender_domain and reply_domain and registrable_domain(sender_domain) != registrable_domain(reply_domain):
        indicators.append(
            Indicator(
                name="reply_to_mismatch",
                severity="warning",
                weight=16,
                detail=(
                    f"Reply-To domain {registrable_domain(reply_domain)} differs from "
                    f"From domain {registrable_domain(sender_domain)}."
                ),
            )
        )
    suffix = ""
    if attachment.strip():
        suffix = "." + attachment.strip().lower().rsplit(".", 1)[-1] if "." in attachment else ""
    dangerous = {".html", ".htm", ".iso", ".exe", ".js", ".vbs", ".scr", ".bat", ".cmd", ".lnk", ".apk", ".msi", ".svg"}
    if suffix in dangerous:
        indicators.append(
            Indicator(
                name="dangerous_attachment_name",
                severity="critical",
                weight=28,
                detail=f"The attachment name ends with {suffix}. Only the filename is checked, not the file contents.",
            )
        )
    return indicators


_MAILBOX = re.compile(r"^\s*(?P<name>.*?)\s*<(?P<addr>[^>]+)>\s*$")


def _split_mailbox(value: str) -> tuple[str, str]:
    text = (value or "").strip()
    match = _MAILBOX.match(text)
    if match:
        return match.group("name").strip().strip('"'), match.group("addr").strip()
    if "@" in text:
        return "", text
    return text, ""


def _domain(address: str) -> str:
    if "@" not in address:
        return ""
    return address.rsplit("@", 1)[-1].strip().strip(">").lower()
