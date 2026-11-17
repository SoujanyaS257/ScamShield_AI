"""Bundled scanner examples. They are deliberately not copied verbatim from the training corpus."""

from __future__ import annotations

SMS_SAFE = (
    "Arun, the design critique moved to 4 pm in studio B. I already printed the boards."
)
SMS_FRAUD = (
    "Bank security desk: your account is frozen. Read me the one-time passcode we just sent "
    "or the debit card will be blocked within 10 minutes."
)
WHATSAPP_FRAUD = (
    "Forwarded many times. A government benefit is waiting. Share this with all your contacts "
    "and pay the processing fee by wire transfer today."
)
EMAIL_BODY = (
    "We detected a new withdrawal. Confirm your identity and send the password within 30 minutes "
    "or the mailbox will be disabled."
)
EMAIL_SENDER = "PayPal Security <alert@paypa1-secure.example>"
EMAIL_SUBJECT = "Wallet limited — verify now"
TRANSCRIPT_FRAUD = (
    "This is the income tax officer assigned to your file. Do not hang up. A warrant is being "
    "prepared unless you buy gift cards and read the codes to me now."
)
URL_SAFE = "https://www.python.org/downloads/"
URL_FRAUD = "http://paypa1.com/login/verify"
URL_SHORT = "https://bit.ly/scamshield-demo"
QR_PAYLOAD = "http://kyc-update-sbi.example/login"
IMAGE_TEXT = (
    "Your mailbox is suspended. Send the one-time passcode and pay the release fee today."
)
VOICE_SCRIPT = (
    "This is the bank fraud desk. Do not hang up. Read me the verification code that was just sent to your phone."
)
