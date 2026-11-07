import re, json, os

BASE = os.path.dirname(os.path.abspath(__file__))
QE = os.path.join(BASE, "audit_reference", "dvg_test.jsonl")
ANS = os.path.join(BASE, "results_full", "dvgb_hard_full", "ans_%s_explicit_873.jsonl")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def extract_explicit(answer):
    m = re.search(r"<explicit>(.*?)(?:</explicit>|<answer>|\Z)", answer or "", re.S)
    return m.group(1).strip() if m else ""


def tokens(t):
    return set(re.findall(r"[a-z0-9]+", (t or "").lower()))


def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# 方位词：人工标注里的核心模式
DIRW = re.compile(
    r"\b(left|right|top|bottom|upper|lower|center|centre|middle|corner|"
    r"slightly|above|below|near|beside|along|between|edge)\b", re.I
)
# 严格方位短语：slightly to the X
PHRASE = re.compile(r"slightly to the (?:upper |lower )?(?:left|right)|in the (?:middle|center)", re.I)


def stats(names, lens, hits_dir, hits_phrase, jacs=None, zero=None):
    n = len(lens)
    out = {
        "n": n,
        "len_mean": round(sum(lens) / n, 1),
        "len_median": sorted(lens)[n // 2],
        "dir_rate": round(100 * hits_dir / n, 1),
        "phrase_rate": round(100 * hits_phrase / n, 1),
    }
    if jacs is not None:
        out["jac_mean"] = round(sum(jacs) / n, 4)
        out["jac_median"] = round(sorted(jacs)[n // 2], 4)
        out["jac_ge_09"] = round(100 * sum(1 for x in jacs if x >= 0.9) / n, 2)
        out["jac_ge_05"] = round(100 * sum(1 for x in jacs if x >= 0.5) / n, 2)
        out["jac_zero"] = round(100 * sum(1 for x in jacs if x == 0) / n, 1)
    return out


def main():
    qe_rows = load(QE)
    # IDs repeat independently; align by the full sample key.
    src = {}
    for r in qe_rows:
        key = (r["dataset"], r["image_id"], r["question_id"])
        if key in src:
            raise ValueError(f"重复测试样本键: {key}")
        src[key] = r

    report = {}

    # ---- 人工 ----
    human = [r.get("question_e") or "" for r in qe_rows]
    h = stats("human", [len(x) for x in human],
              sum(1 for x in human if DIRW.search(x)),
              sum(1 for x in human if PHRASE.search(x)))
    report["human"] = h

    # ---- 模型 ----
    for g in ["g2", "g6"]:
        rows = load(ANS % g)
        pairs = []
        seen = set()
        for r in rows:
            key = (r["dataset"], r["image_id"], r["question_id"])
            source = src.get(key)
            if source is None or key in seen:
                raise ValueError(f"{g} 输出样本键缺失或重复: {key}")
            if r.get("question") != source.get("question_e"):
                raise ValueError(f"{g} explicit 查询文本不匹配: {key}")
            seen.add(key)
            pairs.append((extract_explicit(r.get("answer")), source.get("question_e") or ""))
        if seen != set(src):
            raise ValueError(f"{g} 输出样本集合与测试集不一致")

        ex = [p[0] for p in pairs]
        jacs = [jaccard(a, b) for a, b in pairs]
        st = stats(g, [len(x) for x in ex],
                   sum(1 for x in ex if DIRW.search(x)),
                   sum(1 for x in ex if PHRASE.search(x)),
                   jacs)
        st["n_paired"] = len(pairs)
        report[g] = st

    # ---- 交叉：只在人工含方位词的样本上比 ----
    cross = {}
    for g in ["g2", "g6"]:
        rows = load(ANS % g)
        sub_j = []
        for r in rows:
            key = (r["dataset"], r["image_id"], r["question_id"])
            gt = src[key].get("question_e") or ""
            if DIRW.search(gt):
                sub_j.append(jaccard(extract_explicit(r.get("answer")), gt))
        cross[g] = {
            "n": len(sub_j),
            "jac_mean": round(sum(sub_j) / len(sub_j), 4) if sub_j else 0,
            "jac_zero": round(100 * sum(1 for x in sub_j if x == 0) / len(sub_j), 1) if sub_j else 0,
        }
    report["cross_dir_only"] = cross

    out = os.path.join(BASE, "results_full", "recompute", "explicit_quality.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    print("=" * 78)
    print("%-8s %6s %8s %8s %9s %9s %8s %8s %7s" %
          ("组", "n", "均长", "中位", "含方位词", "含方位短语", "Jac均值", "≥0.9", "=0"))
    print("-" * 78)
    for k, lab in [("human", "人工"), ("g2", "g2"), ("g6", "g6")]:
        d = report[k]
        print("%-8s %6d %8.1f %8d %8.1f%% %8.1f%% %8.4f %8.2f %7.1f" %
              (lab, d["n"], d["len_mean"], d["len_median"], d["dir_rate"],
               d["phrase_rate"], d.get("jac_mean", 0), d.get("jac_ge_09", 0),
               d.get("jac_zero", 0)))
    print()
    print("仅人工含方位词的子集上比 Jaccard：")
    for g in ["g2", "g6"]:
        c = cross[g]
        print("  %s  n=%d  Jac均值 %.4f  零重叠 %.1f%%" %
              (g, c["n"], c["jac_mean"], c["jac_zero"]))


if __name__ == "__main__":
    main()
