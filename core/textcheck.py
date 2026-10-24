"""Count a Chinese translation's characters against the Mastered and Learning lists.

From the project folder:
    python3 -m core.textcheck reply.txt                 # a text file holding the model's reply
    python3 -m core.textcheck                           # or the clipboard (macOS)
    python3 -m core.textcheck reply.txt --tagged tagged_story.txt
                                                        # also confirm every <dnt> span from the tagged
                                                        # English story was copied unchanged
    python3 -m core.textcheck reply.txt --max 10        # override translation.target_new_chars

It compares every Chinese character in the reply with data/allowed_chars.json and reports how many
unique characters were used, how many are Mastered / Learning / new, which new characters were added
(with the clause each appeared in), and whether the new-character target held.
Text inside <dnt>...</dnt> tags is protected English: it is not counted, but it must match the tagged
story. A trailing "NOTES:" section is ignored.

The stages import this file: check(), split_reply(), load_allowed(), new_char_target(),
revert_tolerance() and format_report() are shared, so every stage and this script always agree.
Standard library only.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from core.config import BASE, ConfigError, current_config, get, resolve_path

DNT_SPAN = re.compile(r"<dnt>(.*?)</dnt>", re.S)
DNT_TAG = re.compile(r"</?dnt>")
CLAUSE_BREAK = re.compile(r"[。！？，、；：,.!?;:\n]")

# Common characters that exist only in Simplified Chinese. Characters that Traditional text also
# uses (干 云 于 后 里 台 面 系 污 and similar) are deliberately left out, so a hit is a strong sign
# of drift. It is a small built-in list, not a full converter: a clean result doesn't prove the
# text is entirely Traditional.
SIMPLIFIED_ONLY = set(
    "们这说来个时国学对开关会经动现样没过还从点长门问间见书车东马鸟鱼风飞饭带爱儿买卖读写听话语请谢让认识记给总红绿蓝黄"
    "虽为与无业产电脑网刚历压乐兴华钱银铁错难题页顺领须颜双兰实宝将导寻尔层岁岛币师帮应当异弹惊愿态怀恋惯户节药荣获营"
    "虑处备复够头宁宫专尘尽属岂帅汉谁课贝财货购贵费资赛赞趋跃轻载较辆输边达运远连进选递适逻种积称稳穷篮类粮纪约级组细织"
    "终绍结络绝统继续维综缘编缩罗罚联声职肤胜脏脸艺苏荐虫蚁补装觉观规视览计订讨训议讯讲许论设访证评诉词译试诗该详误诸调"
    "谈谱负贡责贤败账质贩贪贫贯贱贴贷贸贺乡习乱争亚亲仅仓价众优伟传伤伦侠侣侦侧债倾偿储兑党养兽内册军农冯决况冻净凉减凑"
    "凤凭击刘则创删别刹剂剑剧劝办务动励劲劳势勋区医协单卢卫厂厅厌县参变叙叠号叹吓吗启吴员响哑哗唤团园围图圆圣场坏块坚坛"
    "坝坞坟坠垄垒垦壳壶夹夸夺奋奖妆妇妈娱婴审宪宽宾寿帐广庆庐库废张归录彦彻径忆忧恳恶悦惨愤慑懒戏战执扩扫扬扰抚报拥择挂"
    "挚挥损换捡据掷摄摆擞敌敛数斋斗断旧显晒晓晕暂术机杀杂权条杨极构枪柜栈档桥梦检楼欢欧毕气汇汤沟沥沦沪泪泽洁洒浅济浓涂"
    "涛润涨渐渔湾湿滚满滞滤滨灭灯灵灾炉烂烛烦烧热爷牵犹狈独狭狮猎猪献玛环玺琐画畅疗疯痒痴瘾盏盐监盖盘睁矿码确礼祸离秃窝"
    "竞笔笼筑筛签简紧纠纯纲纳纵纸纹纺线练绕绘绞绩绪绳绵绽缀缓缚缝缠缴罢聪肃肠肿胀胁脉腾舰舱艰艳芦苍苹茎荡莱萝萧虏蚀蛮衬袄"
    "触讥诚诞询诺谅谊谋谎谐谓谜谣谨贞贼赌赏赔赖赚赠赢赵赶践踪轨轩转轮软轰辅辈辉辑辖辩辽迁迈迟违逊遗邓邮邻郑酿释钉钓钙钞钟"
    "钢钥钦钻铃铜铝铺链销锁锅锋锐键锻镇镜闪闭闯闰闷闸闹闻阀阁阅队阳阴阵阶际陆陈险随隐雏雾韩顶项顾顿颁颂预颇频颗额飘饥饮饱"
    "饼馆驱驳驴驶驻驾骂骄骆骑骗骤鲁鲜鸡鸣鸭鸿鹅鹤鹰麦齐齿龙龟"
)


def is_han(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2A6DF or 0xF900 <= o <= 0xFAFF)


# ---------- shared loading and parsing ----------

def load_allowed(cfg=None, path=None):
    """Read allowed_chars.json (default: data/allowed_chars.json, or paths.data from config)."""
    if path is None:
        data_dir = resolve_path(cfg, "data") if cfg else BASE / "data"
        path = data_dir / "allowed_chars.json"
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run data/pleco_import.py on a Pleco export first.")
    return json.loads(path.read_text(encoding="utf-8"))


def new_char_target(cfg):
    """The new-character target, from translation.target_new_chars (0 or more)."""
    try:
        n = int(get(cfg, "translation.target_new_chars"))
        if n < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError('Set "translation.target_new_chars" in config.json to a whole number (0 or more).')
    return n


def revert_tolerance(cfg):
    """How many characters over the target a translation may be before English reversion runs."""
    try:
        n = int(get(cfg, "translation.revert_tolerance", 10))
        if n < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError('"translation.revert_tolerance" in config.json must be a whole number (0 or more).')
    return n


def split_reply(reply):
    """Separate the translation from an optional NOTES section. Returns (body, notes)."""
    parts = re.split(r"^\s*NOTES:", reply, maxsplit=1, flags=re.M)
    body = parts[0].strip()
    notes = parts[1].strip() if len(parts) > 1 else ""
    if notes.strip(" .\"'").lower() == "none":
        notes = ""
    return body, notes


def strip_dnt_tags(text):
    """Remove <dnt> and </dnt> but keep the text between them."""
    return DNT_TAG.sub("", text)


def extract_dnt(text):
    """The text of every <dnt>...</dnt> span, in order."""
    return [s.strip() for s in DNT_SPAN.findall(text)]


# ---------- the check ----------

BREAK_CHARS = "。！？，、；：,.!?;:\n"
_PROTECT = {c: chr(0xE000 + i) for i, c in enumerate(BREAK_CHARS)}     # private-use stand-ins
_RESTORE = {v: k for k, v in _PROTECT.items()}


def _context_text(body):
    """The reply as the learner will read it (<dnt> tags removed), with punctuation inside protected
    text such as "U.S." hidden so it doesn't split a clause in two. Pass the result to _context()."""
    return DNT_SPAN.sub(lambda m: "".join(_PROTECT.get(c, c) for c in m.group(1)), body)


def _ascii_word(ch):
    return ch.isascii() and ch.isalnum()


def _context(text, ch, width=40):
    """The clause of text (from _context_text) that first contains ch, shortened to about width
    characters around it. A shortened end never cuts an English word in half and is marked "…"."""
    for clause in CLAUSE_BREAK.split(text):
        if ch not in clause:
            continue
        clause = clause.strip()
        start, end = 0, len(clause)
        if len(clause) > width:
            i = clause.index(ch)
            start = max(0, min(i - width // 2, len(clause) - width))
            end = start + width
            while start > 0 and start < i and _ascii_word(clause[start - 1]) and _ascii_word(clause[start]):
                start += 1                                  # don't begin in the middle of a word
            while end < len(clause) and end > i + 1 and _ascii_word(clause[end - 1]) and _ascii_word(clause[end]):
                end -= 1                                    # don't end in the middle of a word
        snippet = clause[start:end].strip()
        snippet = ("…" if start > 0 else "") + snippet + ("…" if end < len(clause) else "")
        return "".join(_RESTORE.get(c, c) for c in snippet)
    return ""


def check(reply, allowed, max_new, expected_dnt=None, strict_simplified=False):
    """Check one model reply. Pure: no printing, no exiting.

    reply             the model's full reply (translation plus optional NOTES)
    allowed           the dict from load_allowed()
    max_new           the target for different characters outside Mastered and Learning
    expected_dnt      the <dnt> spans of the tagged English story, or None when the story was not
                      tagged (spans are then not verified and stray Latin letters only warn)
    strict_simplified treat simplified-only characters as a failure instead of a warning

    Returns a dict. "passed" is True when the new-character count is within max_new and there are no
    problems. "score" ranks attempts (lower is better): (new characters outside the preferred pool,
    new characters, problems, new-character occurrences).
    """
    mastered, learning = set(allowed["mastered"]), set(allowed["learning"])
    pool = {u["char"] for u in allowed["unknown"]}

    body, notes = split_reply(reply)
    text = strip_dnt_tags(body)                                  # what the learner will see
    found = extract_dnt(body)
    outside_tags = strip_dnt_tags(DNT_SPAN.sub("", body))        # everything that must be Chinese
    counts = Counter(ch for ch in outside_tags if is_han(ch))

    groups = {"mastered": [], "learning": [], "unknown": []}
    for c in counts:
        groups["mastered" if c in mastered else "learning" if c in learning else "unknown"].append(c)
    occ = {k: sum(counts[c] for c in v) for k, v in groups.items()}
    total = sum(counts.values())
    unknown = sorted(groups["unknown"], key=lambda c: (c not in pool, -counts[c], c))   # preferred-list characters first
    in_pool = [c for c in unknown if c in pool]
    outside = [c for c in unknown if c not in pool]
    ctx_text = _context_text(body)
    new_details = [{"char": c, "count": counts[c], "in_pool": c in pool, "context": _context(ctx_text, c)}
                   for c in unknown]

    problems, warnings = [], []
    if not counts:
        problems.append("the reply has no Chinese characters in it")

    if expected_dnt is not None:
        want, got = Counter(s.strip() for s in expected_dnt), Counter(found)
        missing = sorted((want - got).elements())
        surplus = sorted((got - want).elements())
        misused = [s for s in surplus if any(is_han(c) for c in s)]
        extra = [s for s in surplus if s not in misused]
        if missing:
            problems.append("protected text was missing or changed: " + ", ".join(f'"{s}"' for s in missing))
        if misused:
            problems.append("Chinese text was wrapped in <dnt> tags, which is only for English: "
                            + ", ".join(f'"{s}"' for s in misused))
        if extra:
            warnings.append("extra <dnt> spans that were not in the story: " + ", ".join(f'"{s}"' for s in extra))
    if body.count("<dnt>") != body.count("</dnt>"):
        warnings.append("unbalanced <dnt> tags in the reply")

    latin = re.findall(r"[A-Za-z]+", outside_tags)
    if latin:
        warnings.append("English letters outside <dnt> tags (something left untranslated?): "
                        + ", ".join(latin[:6]) + (" ..." if len(latin) > 6 else ""))

    simplified = sorted(c for c in counts if c in SIMPLIFIED_ONLY)
    if simplified:
        msg = "simplified-only characters (Traditional was required): " + "".join(simplified)
        (problems if strict_simplified else warnings).append(msg)

    passed = len(unknown) <= max_new and not problems
    score = (len(outside), len(unknown), len(problems), sum(counts[c] for c in unknown))
    pct = lambda k: round(100 * occ[k] / total) if total else 0
    return {
        "body": body, "text": text, "notes": notes, "dnt_found": found,
        "total_chars": total, "unique": len(counts),
        "mastered": groups["mastered"], "learning": groups["learning"], "unknown": unknown,
        "in_pool": in_pool, "outside_pool": outside, "new_details": new_details,
        "percent": {k: pct(k) for k in occ},
        "simplified": simplified, "problems": problems, "warnings": warnings,
        "limit": max_new, "passed": passed, "score": score,
    }


# ---------- report ----------

def format_report(r):
    """A plain-text report of one check() result."""
    lines = [f"Checked {r['total_chars']} Chinese characters in the reply.", "",
             f"Total unique characters used: {r['unique']}",
             f"  Mastered: {len(r['mastered'])}",
             f"  Learning: {len(r['learning'])}",
             f"  New:      {len(r['unknown'])}  "
             f"({len(r['in_pool'])} from the preferred list, {len(r['outside_pool'])} outside it)",
             "", f"By occurrence: {r['percent']['mastered']}% Mastered, {r['percent']['learning']}% Learning, "
                 f"{r['percent']['unknown']}% New",
             "", "New characters (not in your Mastered or Learning lists):"]
    if r["new_details"]:
        for d in r["new_details"]:
            where = "preferred list" if d["in_pool"] else "outside the list"
            lines.append(f"  {d['char']}  x{d['count']}  ({where})  in \"{d['context']}\"")
    else:
        lines.append("  none")
    if r["dnt_found"]:
        lines += ["", "Protected English (<dnt>): " + ", ".join(f'"{s}"' for s in r["dnt_found"])]
    if r["notes"]:
        lines += ["", "Model's notes: " + " ".join(r["notes"].split())]
    for p in r["problems"]:
        lines += ["", "Problem: " + p]
    for w in r["warnings"]:
        lines += ["", "Warning: " + w]
    lines += ["", f"New-character target ({r['limit']}): {len(r['unknown'])} used, "
                  + ("WITHIN TARGET" if r["passed"] else "OVER TARGET")]
    return "\n".join(lines)


# ---------- command line ----------

def read_input(path):
    if path:
        try:
            return Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"Could not read {path}: {e}")
    try:
        return subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        sys.exit("Could not read the clipboard. Pass a file instead: python3 -m core.textcheck reply.txt")


def main():
    ap = argparse.ArgumentParser(description="Check a Chinese translation against the character lists.")
    ap.add_argument("reply", nargs="?", help="text file with the model's reply (default: the clipboard)")
    ap.add_argument("--tagged", help="the English story with <dnt> tags, to confirm protected text was kept")
    ap.add_argument("--max", type=int, help="new-character target (default: translation.target_new_chars)")
    ap.add_argument("--allowed", help="path to allowed_chars.json (default: data/allowed_chars.json)")
    ap.add_argument("--strict-simplified", action="store_true", help="fail on simplified-only characters")
    args = ap.parse_args()

    try:
        cfg = current_config()
        allowed = load_allowed(cfg, args.allowed)
        max_new = args.max if args.max is not None else new_char_target(cfg)
    except (RuntimeError, ConfigError) as e:
        sys.exit(str(e))
    expected = None
    if args.tagged:
        try:
            expected = extract_dnt(Path(args.tagged).expanduser().read_text(encoding="utf-8"))
        except OSError as e:
            sys.exit(f"Could not read {args.tagged}: {e}")

    result = check(read_input(args.reply), allowed, max_new, expected, args.strict_simplified)
    if not result["total_chars"]:
        sys.exit("No Chinese characters found in the text.")
    print(format_report(result))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
