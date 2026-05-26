import pytest
from md_hybrid_search.tokenizer import MeCabTokenizer, RegexTokenizer, get_tokenizer

def test_regex_tokenizer():
    tokenizer = RegexTokenizer()
    text = "  Hello   WORLD! \n New line. "
    normalized = tokenizer.normalize(text)
    # _JP_LATIN_TOKEN_RE = r"[\w\u3040-\u30ff\u3400-\u9fff]+"
    # It will strip punctuation like '!' and '.' if they are not considered \w
    # In Python, \w matches alphanumeric + underscore.
    # "hello" "world" "new" "line"
    assert "hello" in normalized
    assert "world" in normalized
    assert "!" not in normalized

def test_get_tokenizer():
    # Reset cache to ensure get_tokenizer runs its logic
    from md_hybrid_search import tokenizer as tokenizer_mod
    tokenizer_mod._tokenizer_cache = None

    tokenizer = get_tokenizer()
    assert tokenizer is not None
    assert hasattr(tokenizer, "normalize")

def test_mecab_tokenizer_fallback(monkeypatch):
    import sys
    from md_hybrid_search import tokenizer as tokenizer_mod

    # Reset cache
    tokenizer_mod._tokenizer_cache = None

    # Force fugashi to be missing
    monkeypatch.setitem(sys.modules, "fugashi", None)

    # MeCabTokenizer() should raise ImportError (forced by sys.modules[fugashi]=None)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        MeCabTokenizer()

    # But get_tokenizer should handle it and fallback to RegexTokenizer
    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, RegexTokenizer)
