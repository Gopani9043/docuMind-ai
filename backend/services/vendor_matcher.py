import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

STRIP_SUFFIXES = [
    "ltd", "gmbh", "inc", "llc", "ag", "analytics",
    "solutions", "group", "co", "corp", "limited",
    "services", "consulting", "technologies", "tech"
]


def normalize(vendor: str) -> str:
    """Strip common suffixes and normalize vendor name."""
    if not vendor:
        return ""
    v = vendor.lower().strip()
    v = re.sub(r'[^\w\s]', '', v)
    words = v.split()
    words = [w for w in words if w not in STRIP_SUFFIXES]
    return " ".join(words).strip()


def similarity(a: str, b: str) -> float:
    """Return similarity score between 0 and 1."""
    na = normalize(a)
    nb = normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def find_matches(
    vendor: str,
    all_vendors: list[str],
    threshold: float = 0.75
) -> list[str]:
    """Find all vendor names similar to the given vendor."""
    matches = []
    for v in all_vendors:
        if v and similarity(vendor, v) >= threshold:
            matches.append(v)
    logger.info(f"Vendor '{vendor}' matched: {matches}")
    return matches