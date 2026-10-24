"""Stage ingest: the raw source (an email body, a pasted page, a text file) -> clean plain text.

    reads   00_source.txt
    writes  01_clean.txt

What it does, in order:
  1. Normalizes line endings, invisible characters and odd spaces.
  2. Turns HTML into text (hidden preheaders, scripts and styles are dropped, links keep only
     their words). Plain text is left as it is.
  3. Removes URLs, "[1]" link references and "[image: ...]" markers.
  4. Joins hard-wrapped plain-text paragraphs back into single lines.
  5. Cuts the newsletter footer (copyright, address, unsubscribe) and removes short boilerplate
     lines anywhere else ("View in browser", "Share on Facebook", "Read more", ...).
  6. Caps the length at stages.ingest.max_chars in config.json (default 12000; 0 turns the cap off).

It only ever removes short lines that look like boilerplate; ordinary sentences are kept. Advertising
copy that sits between articles looks like content and is NOT removed.

To see what it would do to a file without making a job:
    python3 -m stages.ingest samples/messy_newsletter.txt          # cleaned text on stdout, report on stderr
    python3 -m stages.ingest samples/messy_newsletter.txt --max-chars 0
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser

from core import schema

SOURCE_FILE = "00_source.txt"
CLEAN_FILE = "01_clean.txt"
DEFAULT_MAX_CHARS = schema.STAGES["ingest"]["max_chars"][0]          # one default, kept in core/schema.py

# A footer is only cut when it starts in the last 40% of the text and nothing after it looks like
# article prose (no paragraph longer than this).
FOOTER_MIN_POSITION = 0.6
FOOTER_MAX_PARAGRAPH = 600
BOILERPLATE_MAX_LINE = 200
FOOTER_MARKER_MAX_LINE = 300


# --------------------------------------------------------------------------- patterns

_INVISIBLE = re.compile("[​‌‍⁠﻿­͏᠎‎‏‪-‮]")
_SPACES = re.compile("[  -   　\t\f\v]")

_URL_MARKDOWN = re.compile(r"\[([^\]]*)\]\(\s*(?:https?://|mailto:|www\.)[^)]*\)", re.I)
_URL_ANGLE = re.compile(r"<\s*(?:https?://|mailto:|www\.)[^>\s]*\s*>", re.I)
_URL_BARE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
_IMAGE_MARK = re.compile(r"\[\s*(?:image|img|photo|picture|logo|video|gif)\b[^\]]{0,100}\]", re.I)
_REF_MARK = re.compile(r"\s?\[\d{1,3}\]")

_HTML_HINT = re.compile(
    r"<\s*(?:!doctype|html|head|body|div|p|br|table|tr|td|span|a|img|h[1-6]|ul|ol|li|style|center|"
    r"font|b|i|strong|em)\b[^>]*>", re.I)

# Lines that mark the start of a newsletter footer. Kept strict on purpose: a sentence in an
# article about privacy must not look like one.
_FOOTER_MARKERS = [
    ("unsubscribe", re.compile(r"\bunsubscribe\b", re.I)),
    ("receiving this", re.compile(r"\byou(?:['’]re| are) (?:receiving|getting) this\b", re.I)),
    ("received this", re.compile(r"\byou (?:received|got) this (?:e-?mail|message|newsletter)\b", re.I)),
    ("manage preferences", re.compile(r"\b(?:manage|update|change|edit) (?:your )?(?:e-?mail )?(?:preferences|subscription)\b", re.I)),
    ("email was sent", re.compile(r"\bthis (?:e-?mail|message) (?:was|is|has been) sent (?:to|by)\b", re.I)),
    ("rights reserved", re.compile(r"\ball rights reserved\b", re.I)),
    ("copyright", re.compile(r"^\s*(?:copyright\s*)?(?:©|\(c\))|^\s*copyright\s*\d{4}", re.I)),
]

# Short boilerplate lines removed wherever they appear.
_BOILERPLATE = _FOOTER_MARKERS + [
    ("view in browser", re.compile(r"\b(?:view|read|open|see)\b.{0,40}\b(?:in|on)\b.{0,15}\bbrowser\b", re.I)),
    ("view online", re.compile(r"\b(?:view|read) (?:it |this |the )?(?:e-?mail |newsletter |message )?online\b", re.I)),
    ("trouble viewing", re.compile(r"\bhaving trouble (?:viewing|reading|seeing)\b", re.I)),
    ("forward", re.compile(r"\bforward (?:this|it|to)\b.{0,40}\b(?:friend|colleague|someone|e-?mail)\b", re.I)),
    ("address book", re.compile(r"\badd\b.{0,50}\bto your (?:address book|contacts|safe senders)\b", re.I)),
    ("stop receiving", re.compile(r"\b(?:to stop receiving|no longer (?:wish|want) to (?:receive|get)|if you (?:no longer|don['’]t|do not) (?:wish|want))\b", re.I)),
    ("why this email", re.compile(r"\bwhy did i get this\b", re.I)),
    ("subscribe prompt", re.compile(r"\b(?:subscribe|sign up) (?:now|today|here|for|to (?:our|the|this))\b", re.I)),
    ("click here", re.compile(r"^\s*(?:click|tap) here\b", re.I)),
    ("read more", re.compile(r"^\s*(?:read|learn|see|view) (?:more|the full (?:story|article))\b.{0,30}$", re.I)),
    ("skip link", re.compile(r"^\s*skip to (?:main )?content\b", re.I)),
    ("ad label", re.compile(r"^\s*(?:advertisement|sponsored(?: content)?|sponsor|partner content|promoted)\s*:?\s*$", re.I)),
    ("follow us", re.compile(r"^\s*(?:follow|find|connect with|join) (?:us|me)\b", re.I)),
]
_PRIVACY_TERMS = re.compile(r"\b(?:privacy policy|terms of (?:service|use)|terms (?:&|and) conditions)\b", re.I)
_SOCIAL_NAMES = re.compile(
    r"\b(?:facebook|twitter|instagram|linkedin|youtube|tiktok|whatsapp|threads|bluesky|mastodon|pinterest|reddit|email|web)\b", re.I)

# Words that may sit around the network names in a row of social links ("Share on Facebook | Twitter").
_SOCIAL_FILLER = re.compile(
    r"\b(?:and|share|tweet|post|follow|find|us|on|via|with|connect|join|like|this|story|article|page|our)\b", re.I)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•·▪‣◦–—]|\d{1,2}[.)])\s+")
_SENTENCE_END = re.compile(r"[.!?][\"”’')]*$")


# --------------------------------------------------------------------------- report

@dataclass
class CleanReport:
    before: int = 0
    after: int = 0
    was_html: bool = False
    removed: list = field(default_factory=list)       # (reason, line)
    footer_start: str = ""
    footer_lines: int = 0
    truncated_to: int = 0                             # characters kept, when the cap applied

    def summary(self):
        parts = [f"{self.before:,} -> {self.after:,} characters"]
        if self.was_html:
            parts.append("HTML converted to text")
        if self.removed:
            parts.append(f"removed {len(self.removed)} boilerplate line{'s' if len(self.removed) != 1 else ''}")
        if self.footer_lines:
            parts.append(f"cut a {self.footer_lines}-line footer")
        if self.truncated_to:
            parts.append(f"CUT TO THE FIRST {self.truncated_to:,} CHARACTERS (stages.ingest.max_chars)")
        text = "; ".join(parts) + "."
        if self.before > 500 and self.after < 0.25 * self.before and not self.truncated_to:
            text += " WARNING: most of the text was removed; check 01_clean.txt."
        return text

    def detail(self, limit=40):
        lines = [self.summary()]
        if self.footer_lines:
            lines.append(f'Footer cut from: "{self.footer_start[:100]}"')
        for reason, text in self.removed[:limit]:
            lines.append(f"  removed [{reason}] {text[:110]}")
        if len(self.removed) > limit:
            lines.append(f"  ... and {len(self.removed) - limit} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------- HTML to text

_SKIP_TAGS = {"script", "style", "head", "noscript", "svg", "template", "title"}
_VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "wbr"}
_LINE_TAGS = {"br", "div", "li", "tr", "td", "th", "hr"}
_PARA_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "table", "ul", "ol",
              "section", "article", "header", "footer", "pre"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|mso-hide\s*:\s*all|max-height\s*:\s*0(?:px)?\s*(?:;|$)", re.I)


def looks_like_html(text):
    return len(_HTML_HINT.findall(text)) >= 2


class _HtmlToText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
        self.skip = 0
        self.hide_at = None            # stack depth of the element that started hidden content

    @property
    def quiet(self):
        return self.skip > 0 or self.hide_at is not None

    def _break(self, tag):
        if tag in _PARA_TAGS:
            self.parts.append("\n\n")
        elif tag in _LINE_TAGS:
            self.parts.append("\n")

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID_TAGS:
            self.stack.append(tag)
            if tag in _SKIP_TAGS:
                self.skip += 1
            if self.hide_at is None:
                style = dict(attrs).get("style") or ""
                if _HIDDEN_STYLE.search(style) or any(k == "hidden" for k, _ in attrs):
                    self.hide_at = len(self.stack)
        if not self.quiet:
            self._break(tag)

    def handle_startendtag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            return                       # <svg/> and the like hold nothing
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS or tag not in self.stack:
            return
        while self.stack:
            top = self.stack.pop()
            if top in _SKIP_TAGS:
                self.skip -= 1
            if self.hide_at is not None and len(self.stack) < self.hide_at:
                self.hide_at = None
            if top == tag:
                break
        if not self.quiet:
            self._break(tag)

    def handle_data(self, data):
        if not self.quiet:
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(source):
    parser = _HtmlToText()
    parser.feed(source)
    parser.close()
    return "".join(parser.parts)


# --------------------------------------------------------------------------- cleaning steps

def _normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    return _SPACES.sub(" ", text)


def _clean_line(line):
    """Remove URLs and link markers from one line. Returns (text, dropped_as_link_label)."""
    had_url = False
    line = _URL_MARKDOWN.sub(r"\1", line)
    line, n1 = _URL_ANGLE.subn("", line)
    line, n2 = _URL_BARE.subn("", line)
    had_url = bool(n1 or n2)
    line = _IMAGE_MARK.sub("", line)
    line = _REF_MARK.sub("", line)
    line = re.sub(r" {2,}", " ", line).strip()
    if had_url and line.endswith(":") and len(line) <= 60:
        return "", True
    return line, False


def _has_content(line):
    return any(ch.isalnum() for ch in line)


def _blocks(lines):
    """Group consecutive non-empty lines. A line with no letters or digits is a separator."""
    blocks, current = [], []
    for line in lines:
        if line and _has_content(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _unwrap(block):
    """Join a hard-wrapped plain-text paragraph into one line, when it clearly is one."""
    if len(block) < 2:
        return block
    body = block[:-1]
    if any(_LIST_MARKER.match(line) for line in block):
        return block
    if not all(45 <= len(line) <= 110 for line in body) or len(block[-1]) > 110:
        return block
    if sum(1 for line in body if _SENTENCE_END.search(line)) / len(body) >= 0.5:
        return block
    return [" ".join(block)]


def _flatten(blocks):
    lines = []
    for block in blocks:
        if lines:
            lines.append("")
        lines.extend(block)
    return lines


def _is_social_row(line):
    if not _SOCIAL_NAMES.search(line):
        return False
    rest = _SOCIAL_NAMES.sub("", line)
    rest = _SOCIAL_FILLER.sub("", rest)
    rest = re.sub(r"[\s|·•/,\-–—:&]", "", rest)
    return not rest


def _boilerplate_reason(line):
    if len(line) > BOILERPLATE_MAX_LINE:
        return None
    for label, pattern in _BOILERPLATE:
        if pattern.search(line):
            return label
    if _PRIVACY_TERMS.search(line) and len(line) <= 100 and (re.search(r"[|·•]", line) or len(line.split()) <= 6):
        return "privacy/terms link"
    if _is_social_row(line):
        return "social links"
    return None


def _cut_footer(lines, report):
    total = sum(len(line) + 1 for line in lines) or 1
    offset = 0
    for i, line in enumerate(lines):
        if line and len(line) <= FOOTER_MARKER_MAX_LINE and any(p.search(line) for _, p in _FOOTER_MARKERS):
            if offset / total >= FOOTER_MIN_POSITION:
                rest = lines[i:]
                paragraphs = [len(b) for b in ("\n".join(rest).split("\n\n"))]
                if max(paragraphs, default=0) <= FOOTER_MAX_PARAGRAPH:
                    report.footer_start = line
                    report.footer_lines = sum(1 for x in rest if x)
                    return lines[:i]
            break                        # the first marker decides; an early one is just a line
        offset += len(line) + 1
    return lines


def _drop_boilerplate(lines, report):
    kept = []
    for line in lines:
        reason = _boilerplate_reason(line) if line else None
        if reason:
            report.removed.append((reason, line))
        else:
            kept.append(line)
    return kept


def _truncate(text, max_chars, report):
    if not max_chars or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = window.rfind("\n\n")
    if cut < max_chars * 0.5:
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        cut = cut + 1 if cut >= max_chars * 0.5 else window.rfind(" ")
    if cut <= 0:
        cut = max_chars
    kept = text[:cut].rstrip()
    report.truncated_to = len(kept)
    return kept


def clean(source, max_chars=DEFAULT_MAX_CHARS):
    """Pure function: raw source text -> (clean text, CleanReport)."""
    report = CleanReport(before=len(source))
    text = _normalize(source)
    report.was_html = looks_like_html(text)
    if report.was_html:
        text = _normalize(html_to_text(text))

    lines = []
    for raw in text.split("\n"):
        line, dropped = _clean_line(raw.strip())
        if dropped:
            report.removed.append(("link label", raw.strip()))
        lines.append(line)

    blocks = _blocks(lines)
    if not report.was_html:
        blocks = [_unwrap(b) for b in blocks]
    lines = _flatten(blocks)
    lines = _cut_footer(lines, report)
    lines = _drop_boilerplate(lines, report)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = _truncate(text, max_chars, report)
    report.after = len(text)
    return text, report


# --------------------------------------------------------------------------- the stage

def _max_chars(cfg):
    return schema.stage_opts(cfg, "ingest")["max_chars"]


def run(cfg, job):
    source = job.read(SOURCE_FILE)
    if not source.strip():
        raise RuntimeError("The source text is empty.")
    text, report = clean(source, _max_chars(cfg))
    for reason, line in report.removed:
        job.log(f"ingest removed [{reason}]: {line[:120]}")
    if report.footer_lines:
        job.log(f"ingest cut footer starting at: {report.footer_start[:120]}")
    if not text:
        raise RuntimeError("Nothing was left after cleaning the source. Is it only boilerplate? "
                           "Check 00_source.txt and run: python3 -m stages.ingest <file> to see what is removed.")
    job.write(CLEAN_FILE, text + "\n")
    job.set_meta(ingest_cut_to=report.truncated_to)          # 0 unless the article was cut at stages.ingest.max_chars
    return report.summary()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m stages.ingest",
                                 description="Show what ingest does to a file. Makes no job.")
    ap.add_argument("file", nargs="?", help="text or HTML file (default: standard input)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"length cap (default {DEFAULT_MAX_CHARS}; 0 = none)")
    args = ap.parse_args(argv)
    if args.file:
        try:
            with open(args.file, encoding="utf-8-sig") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.exit(f"Could not read {args.file}: {e}")
    else:
        source = sys.stdin.read()
    text, report = clean(source, args.max_chars)
    print(text)
    print(report.detail(), file=sys.stderr)


if __name__ == "__main__":
    main()
