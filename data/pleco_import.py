import json, re, sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

MASTERED_SCORE = 2_000    # natural gap in your data; calibrate further
NEWEST_FIRST = True       # is the first digit of `history` the latest review?
WORD_CAP = "learning"     # max status a word card can give its characters
RANK = {"unknown": 0, "learning": 1, "mastered": 2}
UNKNOWN_POOL_SIZE = 200   # how many "unknown" candidates go into allowed_chars.json

def is_han(ch):
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"

def card_status(si):
    if si is None or int(si.get("reviewed", 0)) == 0:
        return "unknown"                       # in the deck, never studied
    hist = si.get("history", "")
    last = int(hist[0 if NEWEST_FIRST else -1]) if hist else 0
    if last <= 3:                              # most recent answer was a miss
        return "learning"
    return "mastered" if int(si.get("score", 0)) >= MASTERED_SCORE else "learning"

def hsk_level(cats):
    lv = [int(c.split("Level ")[1].split("-")[0])
          for c in cats if c.startswith("HSK 3.0/Level ")]
    return min(lv) if lv else None

BAND_RE = re.compile(r"^Top \d+ Single Character Words/(\d+)$")

def top_band(cats):
    """Lowest hundreds set a character belongs to, e.g. 100, 200, 300 (None if it has none)."""
    bands = []
    for c in cats:
        m = BAND_RE.match(c or "")
        if m:
            bands.append(int(m.group(1)))
    return min(bands) if bands else None

def build_allowed(exported, chars):
    mastered = [r["trad"] for r in chars.values() if r["status"] == "mastered"]
    learning = [r["trad"] for r in chars.values() if r["status"] == "learning"]
    unknown = [(top_band(r["cats"]), r["trad"]) for r in chars.values() if r["status"] == "unknown"]
    # Lowest hundreds set first (100, then 200, then 300, ...); characters with no such
    # category go last. The sort is stable, so ties keep the order of the Pleco export.
    unknown.sort(key=lambda t: (t[0] is None, t[0] or 0))
    pool = unknown[:UNKNOWN_POOL_SIZE]
    return {
        "exported": exported,
        "mastered": "".join(mastered),
        "learning": "".join(learning),
        "unknown": [{"char": ch, "band": band} for band, ch in pool],
        "counts": {"mastered": len(mastered), "learning": len(learning),
                   "unknown_offered": len(pool), "unknown_total": len(unknown)},
    }

def parse(path):
    root = ET.parse(path).getroot()
    chars = {}
    for card in root.iter("card"):
        h = {e.get("charset"): (e.text or "") for e in card.iter("headword")}
        sc = h.get("sc", "")
        tc = h.get("tc", sc)
        pron = card.findtext(".//pron", "")
        defn = card.findtext(".//defn", "")
        si = card.find("scoreinfo")
        status = card_status(si)
        is_word = len(sc) > 1
        if is_word and RANK[status] > RANK[WORD_CAP]:
            status = WORD_CAP
        cats = [] if is_word else [c.get("category") for c in card.iter("catassign")]
        for i, ch in enumerate(sc):
            if not is_han(ch):
                continue
            rec = chars.setdefault(ch, {"status": "unknown", "in_deck": True, "trad": ch,
                                        "pinyin": "", "gloss": "", "score": 0,
                                        "reviewed": 0, "incorrect": 0, "last": 0, "cats": []})
            if RANK[status] > RANK[rec["status"]]:
                rec["status"] = status
            if si is not None:
                rec["score"] = max(rec["score"], int(si.get("score", 0)))
                rec["reviewed"] = max(rec["reviewed"], int(si.get("reviewed", 0)))
                rec["incorrect"] = max(rec["incorrect"], int(si.get("incorrect", 0)))
                rec["last"] = max(rec["last"], int(si.get("lastreviewedtime", 0)))
            if not is_word:
                rec["pinyin"], rec["gloss"] = pron, defn
            if len(tc) == len(sc):
                rec["trad"] = tc[i]
            rec["cats"] = sorted(set(rec["cats"]) | set(cats))
    for rec in chars.values():
        rec["hsk"] = hsk_level(rec["cats"])
    return int(root.get("created", 0)), chars

if __name__ == "__main__":
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    exported, chars = parse(src)
    out = out_dir / f"pleco-snapshot-{exported}.json"
    out.write_text(json.dumps({"type": "pleco_snapshot", "exported": exported,
                               "chars": chars}, ensure_ascii=False))
    print(out.name, dict(Counter(c["status"] for c in chars.values())))
    from make_dashboard import write_dashboard
    write_dashboard({"exported": exported, "chars": chars},
                    Path(__file__).resolve().parent / "dashboard.html")
    allowed = build_allowed(exported, chars)
    (Path(__file__).resolve().parent / "allowed_chars.json").write_text(
        json.dumps(allowed, ensure_ascii=False, indent=1), encoding="utf-8")
    print("allowed_chars.json", allowed["counts"])
