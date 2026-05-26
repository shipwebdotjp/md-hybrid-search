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
    tokenizer = get_tokenizer()
    assert tokenizer is not None
    assert hasattr(tokenizer, "normalize")

def test_mecab_tokenizer_fallback():
    # Since fugashi is not installed in this environment, MeCabTokenizer() should raise ImportError in __init__
    with pytest.raises((ImportError, ModuleNotFoundError)):
        MeCabTokenizer()

    # But get_tokenizer should handle it
    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, RegexTokenizer)
