"""Japanese tokenizer helpers for SQLite FTS5."""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_JP_LATIN_TOKEN_RE = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)


class JanomeTokenizer:
    def __init__(self):
        from janome.tokenizer import Tokenizer

        self._tokenizer = Tokenizer()

    def tokenize(self, text: str) -> List[str]:
        return [t.surface for t in self._tokenizer.tokenize(text) if t.surface.strip()]

    def tokenize_to_string(self, text: str) -> str:
        return " ".join(self.tokenize(text))


class MeCabTokenizer:
    def __init__(self):
        import fugashi

        self._tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> List[str]:
        return [w.surface for w in self._tagger(text) if w.surface.strip()]

    def tokenize_to_string(self, text: str) -> str:
        return " ".join(self.tokenize(text))


class RegexTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in _JP_LATIN_TOKEN_RE.finditer(text) if m.group(0).strip()]

    def tokenize_to_string(self, text: str) -> str:
        return " ".join(self.tokenize(text))


def get_tokenizer():
    try:
        return MeCabTokenizer()
    except Exception as e:
        logger.warning(f"MeCab unavailable, falling back to Janome/regex: {e}")
        try:
            return JanomeTokenizer()
        except Exception as e2:
            logger.warning(f"Janome unavailable, falling back to regex tokenizer: {e2}")
            return RegexTokenizer()
