"""Helpers for the <en>...</en> tags, which mark English put back into a Chinese translation (stage 7).

    <dnt>...</dnt>   English that was never translated (names, acronyms). Helpers are in core/textcheck.py.
    <en>...</en>     English the reversion stage put back in place of Chinese, to avoid a new character.

Both kinds of tag stay in the job's files until stages/package.py removes them, so the judges and the
counter can still tell what is what. Standard library only.
"""
import re

from core import textcheck

EN_SPAN = re.compile(r"<en>(.*?)</en>", re.S)
EN_TAG = re.compile(r"</?en>")


def strip_en_tags(text):
    """Remove <en> and </en> but keep the text between them."""
    return EN_TAG.sub("", text)


def extract_en(text):
    """The text of every <en>...</en> span, in order."""
    return [s.strip() for s in EN_SPAN.findall(text)]


def balanced_en(text):
    """True when every <en> has its </en>, with no nesting."""
    n = len(EN_SPAN.findall(text))
    return text.count("<en>") == text.count("</en>") == n


def drop_en_text(text):
    """The text with the words inside <en> spans removed, so that only Chinese and <dnt> spans are left to
    be counted. A <dnt> span that sits inside an <en> span is kept."""
    def keep_dnt(m):
        return " " + " ".join(f"<dnt>{s}</dnt>" for s in textcheck.extract_dnt(m.group(1))) + " "
    return EN_SPAN.sub(keep_dnt, text)


def strip_all_tags(text):
    """What the reader sees: no <dnt> or <en> tags, the words inside them kept."""
    return strip_en_tags(textcheck.strip_dnt_tags(text))
