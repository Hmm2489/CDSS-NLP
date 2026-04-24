"""Light text normalization applied before the spaCy pipeline.

Keep this conservative: every change shifts character offsets, and
evaluation compares offsets against gold annotations on the *original* text.
For that reason it only does length-preserving fixes for now.
"""

from __future__ import annotations

import re

# Curly quotes break literal matches like "can't breathe".
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    text = text.translate(_QUOTES)
    return re.sub(r"[\t\r\n]", " ", text)
