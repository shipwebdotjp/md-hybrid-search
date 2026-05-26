import re
import logging
from typing import Protocol, Optional, List

logger = logging.getLogger(__name__)

# Pattern for basic CJK + Latin tokenization fallback
_JP_LATIN_TOKEN_RE = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)

class Tokenizer(Protocol):
    def normalize(self, text: str) -> str:
        ...

class MeCabTokenizer:
    def __init__(self):
        import fugashi
        self._tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> List[str]:
        # Lowercase for consistent indexing
        text = text.lower()
        return [w.surface for w in self._tagger(text) if w.surface.strip()]

    def normalize(self, text: str) -> str:
        """Returns a space-separated string of tokens."""
        return " ".join(self.tokenize(text))

class RegexTokenizer:
    def tokenize(self, text: str) -> List[str]:
        # Lowercase for consistent indexing
        text = text.lower()
        return [m.group(0) for m in _JP_LATIN_TOKEN_RE.finditer(text) if m.group(0).strip()]

    def normalize(self, text: str) -> str:
        """Returns a space-separated string of tokens."""
        return " ".join(self.tokenize(text))

_tokenizer_cache: Optional[Tokenizer] = None

def get_tokenizer() -> Tokenizer:
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache

    try:
        _tokenizer_cache = MeCabTokenizer()
        return _tokenizer_cache
    except (ImportError, RuntimeError) as e:
        # Fallback to Regex if fugashi is not installed or MeCab fails with expected errors
        logger.info(f"MeCab (fugashi) unavailable, falling back to RegexTokenizer: {e}")
        _tokenizer_cache = RegexTokenizer()
        return _tokenizer_cache
    except Exception as e:
        # Unexpected error, log and re-raise
        logger.error(f"Unexpected error initializing MeCabTokenizer: {e}")
        raise
