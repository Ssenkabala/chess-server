"""
app_core/profanity.py — a small, standalone profanity filter for Continental
Chat. Word-boundary regex matching, case-insensitive, replaces each matched
word with asterisks of the same length (e.g. "shit" -> "****").

This is a starting list covering common English profanity and slurs. It's a
plain Python list, so it's easy to extend later — just add more entries to
_BLOCKED_WORDS. Matching is whole-word (via \\b), so it won't catch creative
spacing/leetspeak (e.g. "s h i t" or "$hit") — that's a deliberate scope
choice: a heavier fuzzy-matching filter risks false-positiving on legitimate
words (the classic "Scunthorpe problem"), which is worse for a small
community chat than occasionally missing an evasion attempt. Combined with
the 5-messages/day cap and admin visibility into the last 50 messages, this
gives reasonable coverage without over-engineering the parsing.
"""
import re

_BLOCKED_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "cunt", "dick", "piss",
    "crap", "damn", "whore", "slut", "nigger", "nigga", "faggot", "retard",
    "cock", "pussy", "twat", "wanker", "motherfucker", "dumbass", "jackass",
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _BLOCKED_WORDS) + r")\b",
    re.IGNORECASE,
)


def filter_profanity(text: str) -> str:
    """Replace every blocked word with asterisks of the same length."""
    return _PATTERN.sub(lambda m: "*" * len(m.group(0)), text)
