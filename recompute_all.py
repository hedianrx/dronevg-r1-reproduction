#!/usr/bin/env python3
"""
DroneVG-R1 复现 · 10 组统一复算
================================

输入：
  implicit 6 组：ans_g{1..6}.jsonl + summary_g{1..6}.json
  explicit 4 组：dvgb_hard_full/ans_g{1,2,5,6}_explicit_873.jsonl

输出：
  recompute/复算总表.md        人读报告
  recompute/summary_all.json   机读汇总
  recompute/per_sample_g*.jsonl 逐条判定

复算要点：
  1. 统一口径：六类等权（论文口径）为主，样本加权为辅
  2. 无框分类：完全无框 / 有框但IoU<0.5 / 正确
  3. 自定义预测框一致性诊断双口径（非论文 Consistency）：
     - 严格：无框即失败，计入分母
     - 宽松：任一方无框则排除
  4. 增益拆解：区分"无框修复"与"定位变准"两部分的贡献
"""

import json
import os
import re
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_full")
HARD = os.path.join(BASE, "dvgb_hard_full")
OUT = os.path.join(BASE, "recompute")
os.makedirs(OUT, exist_ok=True)

CATEGORIES = ["Security", "Traffic", "Social Activity", "Disaster",
              "Productive Activity", "Sport"]

CLASS_MAP = {
    "security": "Security", "traffic": "Traffic",
    "social": "Social Activity", "social activity": "Social Activity",
    "disaster": "Disaster",
    "productive": "Productive Activity",
    "productive activity": "Productive Activity",
    "sport": "Sport",
}

# 支持负号的框正则
PAT = r"\[(-?[0-9.]+(?:\s*,\s*-?[0-9.]+){3})\]"

PAPER = {
    "g1": 43.78, "g2": 47.93, "g3": 34.82,
    "g4": 31.80, "g5": 51.24, "g6": 52.05,
}
# 论文 Table 3（hard）
PAPER_HARD = {
    "g1": (34.38, 44.70),
    "g2": (None, None),
    "g5": (None, None),
    "g6": (45.66, 47.60),
}

LABELS = {
    "g1": "Qwen2.5-VL-7B + CoT, 0-shot",
    "g2": "Qwen2.5-VL-7B + I2E-CoT, 0-shot",
    "g3": "Qwen2.5-VL-7B + CoT, 2-shot",
    "g4": "Qwen2.5-VL-7B + CoT, 5-shot",
    "g5": "DroneVG-R1 + CoT, 0-shot",
    "g6": "DroneVG-R1 + I2E-CoT, 0-shot",
}


def iou(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = xi2 - xi1, yi2 - yi1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def extract_box(resp):
    """优先 <answer> 内，回退全文第一个"""
    m = re.search(r"<answer>(.*?)</answer>", resp, re.S)
    if m:
        inner = re.findall(PAT, m.group(1))
        if inner:
            return [float(x) for x in inner[0].split(",")], "answer_tag"
    any_box = re.findall(PAT, resp)
    if any_box:
        return [float(x) for x in any_box[0].split(",")], "fallback"
    return None, "none"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sample_key(p):
    return p.get("dataset"), p.get("image_id"), p.get("question_id", p.get("qid"))


def analyze(rows, label):
    """返回逐条判定 + 分类统计"""
    per = []
    for r in rows:
        box, src = extract_box(r.get("answer", ""))
        gt = r.get("bbox")
        ok = box is not None and iou(gt, box) >= 0.5
        cat = CLASS_MAP.get(str(r.get("class", "")).strip().lower(),
                            str(r.get("class", "")))
        per.append(dict(
            qid=r.get("question_id"), image_id=r.get("image_id"),
            dataset=r.get("dataset"),
            cat=cat, box=box, gt=gt, src=src, ok=ok,
            iou=round(iou(gt, box), 4) if box else 0.0,
            has_box=box is not None,
        ))
    return per


def macro_weighted(per):
    """六类等权 + 样本加权 + 无框率"""
    tbl = defaultdict(lambda: [0, 0])
    for p in per:
        c = p["cat"]
        tbl[c][1] += 1
        if p["ok"]:
            tbl[c][0] += 1
    rates = []
    for c in CATEGORIES:
        n = tbl[c][1]
        rates.append(tbl[c][0] / n if n else 0.0)
    tot_n = sum(tbl[c][1] for c in CATEGORIES)
    tot_c = sum(tbl[c][0] for c in CATEGORIES)
    macro = sum(rates) / len(CATEGORIES)
    weighted = tot_c / tot_n if tot_n else 0.0
    n_nobox = sum(1 for p in per if not p["has_box"])
    return dict(macro=macro, weighted=weighted, n=tot_n,
                n_nobox=n_nobox, nobox_rate=n_nobox / tot_n if tot_n else 0,
                per_class={c: (round(tbl[c][0]/tbl[c][1]*100, 2)
                               if tbl[c][1] else None) for c in CATEGORIES},
                n_per_class={c: tbl[c][1] for c in CATEGORIES})


# ============================ 主流程 ============================

print("=" * 72)
print(" 一、Implicit 六组复算")
print("=" * 72)

imp = {}
for g in ["g1", "g2", "g3", "g4", "g5", "g6"]:
    p = os.path.join(BASE, f"ans_{g}.jsonl")
    rows = load_jsonl(p)
    per = analyze(rows, g)
    s = macro_weighted(per)
    imp[g] = dict(per=per, stat=s)
    print(f"{g}: n={s['n']} 等权={s['macro']*100:.2f}% "
          f"加权={s['weighted']*100:.2f}% 无框={s['n_nobox']}({s['nobox_rate']*100:.1f}%)")

print()
print("=" * 72)
print(" 二、Explicit 四组 + 自定义预测框一致性诊断")
print("=" * 72)

hard = {}
hard_questions = load_jsonl(os.path.join(BASE, "dvgb_hard_pilot", "questions_explicit_full.jsonl"))
question_map = {sample_key(q): q for q in hard_questions}
if len(question_map) != len(hard_questions) or len(question_map) != 873:
    raise ValueError("explicit question source must have 873 unique composite keys")
for g in ["g1", "g2", "g5", "g6"]:
    ep = os.path.join(HARD, f"ans_{g}_explicit_873.jsonl")
    erows = load_jsonl(ep)
    eper = analyze(erows, g)
    original_rows = load_jsonl(os.path.join(BASE, f"ans_{g}.jsonl"))
    original_map = {sample_key(r): r for r in original_rows}
    explicit_raw_map = {sample_key(r): r for r in erows}

    # Both question IDs and image IDs can repeat independently.
    ip = {sample_key(p): p for p in imp[g]["per"]}
    ep_map = {sample_key(p): p for p in eper}
    if len(ip) != len(imp[g]["per"]) or len(ep_map) != len(eper):
        raise ValueError(f"{g}: duplicate composite sample key")
    if set(ip) != set(ep_map):
        raise ValueError(f"{g}: implicit/explicit sample keys do not match")
    if set(ip) != set(question_map):
        raise ValueError(f"{g}: model output keys do not match explicit question source")
    for key in question_map:
        q = question_map[key]
        implicit_row = ip[key]
        explicit_row = ep_map[key]
        if implicit_row["gt"] != q["bbox"] or explicit_row["gt"] != q["bbox"]:
            raise ValueError(f"{g}: GT mismatch for {key}")
        original = original_map[key]
        if original["question"] != q["question_i"]:
            raise ValueError(f"{g}: implicit query mismatch for {key}")
        if explicit_raw_map[key]["question"] != q["question"]:
            raise ValueError(f"{g}: explicit query mismatch for {key}")
    common = sorted(set(ip) & set(ep_map))

    # 严格：无框即失败
    both_strict = 0
    cons_strict = 0
    # 宽松：任一方无框则排除
    both_loose = 0
    cons_loose = 0
    n_excluded = 0

    for q in common:
        a, b = ip[q], ep_map[q]
        both_ok = a["ok"] and b["ok"]
        both_strict += int(both_ok)

        pair_has = a["has_box"] and b["has_box"]
        if pair_has:
            same = iou(a["box"], b["box"]) >= 0.5
            both_loose += int(both_ok)
            cons_loose += int(same)
        else:
            n_excluded += 1
        # 严格口径：任一方无框 = 一致性失败
        cons_strict += int(pair_has and iou(a["box"], b["box"]) >= 0.5)

    # 六类等权的 both_correct / consistency
    def macro_rate(fn):
        tbl = defaultdict(lambda: [0, 0])
        for q in common:
            a, b = ip[q], ep_map[q]
            c = a["cat"]
            tbl[c][1] += 1
            tbl[c][0] += int(fn(a, b))
        vals = [tbl[c][0]/tbl[c][1] if tbl[c][1] else 0.0 for c in CATEGORIES]
        return sum(vals) / len(CATEGORIES), tbl

    def f_both(a, b):
        return a["ok"] and b["ok"]

    def f_cons_strict(a, b):
        return a["has_box"] and b["has_box"] and iou(a["box"], b["box"]) >= 0.5

    def f_cons_loose(a, b):
        if not (a["has_box"] and b["has_box"]):
            return None  # 排除
        return iou(a["box"], b["box"]) >= 0.5

    both_macro, _ = macro_rate(f_both)
    cons_s_macro, _ = macro_rate(f_cons_strict)

    # loose 需要单独算（排除后重算分母）
    tbl = defaultdict(lambda: [0, 0])
    for q in common:
        a, b = ip[q], ep_map[q]
        if a["has_box"] and b["has_box"]:
            tbl[a["cat"]][1] += 1
            tbl[a["cat"]][0] += int(iou(a["box"], b["box"]) >= 0.5)
    vals = [tbl[c][0]/tbl[c][1] if tbl[c][1] else 0.0 for c in CATEGORIES]
    cons_l_macro = sum(vals) / len(CATEGORIES)

    hard[g] = dict(
        n=len(common),
        imp_nobox=sum(1 for q in common if not ip[q]["has_box"]),
        exp_nobox=sum(1 for q in common if not ep_map[q]["has_box"]),
        both_macro=both_macro,
        both_count=both_strict,
        cons_strict=cons_s_macro,
        cons_loose=cons_l_macro,
        n_excluded=n_excluded,
        excl_rate=n_excluded / len(common),
    )
    h = hard[g]
    print(f"{g}: n={h['n']} both={h['both_macro']*100:.2f}% | "
          f"框一致性诊断 严格={h['cons_strict']*100:.2f}% 宽松={h['cons_loose']*100:.2f}% | "
          f"排除 {h['n_excluded']} 条({h['excl_rate']*100:.1f}%)")

# ---------- 增益拆解：无框修复 vs 定位变准 ----------
print()
print("=" * 72)
print(" 三、增益拆解：无框修复 vs 定位变准")
print("=" * 72)

def decompose(gA, gB):
    """从 gA 到 gB 的增益，拆成 (原先无框现答对) + (原先有框现答对)"""
    A = {sample_key(p): p for p in imp[gA]["per"]}
    B = {sample_key(p): p for p in imp[gB]["per"]}
    common = set(A) & set(B)
    fix = 0   # A无框 且 B对
    imp_v = 0  # A有框错 且 B对
    lose = 0  # A对 且 B错
    for q in common:
        a, b = A[q], B[q]
        if not a["has_box"] and b["ok"]:
            fix += 1
        elif a["has_box"] and not a["ok"] and b["ok"]:
            imp_v += 1
        elif a["ok"] and not b["ok"]:
            lose += 1
    total = sum(B[q]["ok"] for q in common) - sum(A[q]["ok"] for q in common)
    return dict(n=len(common), gain=total, fixed=fix, improved=imp_v, lost=lose)

pairs = [("g1", "g2", "提示词在基座上"),
         ("g1", "g5", "加载公开 adapter"),
         ("g5", "g6", "提示词在 adapter 模型上")]
decomp = {}
for a, b, name in pairs:
    d = decompose(a, b)
    decomp[f"{a}->{b}"] = dict(name=name, **d)
    print(f"{a}→{b} ({name}): 净增 {d['gain']:+d} 条 "
          f"= 无框修复 {d['fixed']} + 定位改善 {d['improved']} − 丢失 {d['lost']}")

# ---------- 写报告 ----------
lines = []
lines.append("# DroneVG-R1 复现 · 10 组统一复算\n")
lines.append("> 全部数据从原始 `ans_*.jsonl` 重新解析计算，不复用已有 summary。")
lines.append("> 口径：六类等权平均（论文口径）为主。\n")

lines.append("## 一、Implicit 六组（主结果）\n")
lines.append("| 组 | 设置 | n | 论文 | 本次等权 | 差值 | 样本加权 | 无框率 |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for g in ["g1", "g2", "g3", "g4", "g5", "g6"]:
    s = imp[g]["stat"]
    pap = PAPER[g]
    d = s["macro"] * 100 - pap
    lines.append(f"| {g} | {LABELS[g]} | {s['n']} | {pap:.2f}% | "
                 f"**{s['macro']*100:.2f}%** | {d:+.2f} | "
                 f"{s['weighted']*100:.2f}% | {s['nobox_rate']*100:.1f}% |")

lines.append("\n## 二、分类别（六类等权明细）\n")
lines.append("| 组 | " + " | ".join(c.split()[0] if len(c.split()[0]) > 3
                                  else c for c in CATEGORIES) + " |")
lines.append("|---|" + "---|" * 6)
for g in ["g1", "g2", "g5", "g6"]:
    pc = imp[g]["stat"]["per_class"]
    lines.append(f"| {g} | " + " | ".join(
        f"{pc[c]:.1f}" if pc[c] is not None else "—" for c in CATEGORIES) + " |")

lines.append("\n## 三、DVGBench-hard（explicit 配对）\n")
lines.append("| 组 | n | 两查询均正确 | 论文严格准确率 | 预测框 IoU≥0.5（缺框计失败） "
             "| 预测框 IoU≥0.5（缺框排除） | 缺框排除率 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for g in ["g1", "g2", "g5", "g6"]:
    h = hard[g]
    pap_b, pap_c = PAPER_HARD[g]
    pb = f"{pap_b:.2f}%" if pap_b else "—"
    lines.append(f"| {g} | {h['n']} | **{h['both_macro']*100:.2f}%** | {pb} | "
                 f"{h['cons_strict']*100:.2f}% | {h['cons_loose']*100:.2f}% | "
                 f"{h['excl_rate']*100:.1f}% |")
lines.append("\n论文 Table 3 报告的严格准确率：Qwen2.5-VL 7B = 34.38%；"
             "DroneVG-R1 7B = 45.66%。论文 Consistency 数值为 44.70% / 47.60%，"
             "但计算定义未公开，本表不将自定义框一致性诊断与论文数值直接比较。")
lines.append("\n> **自定义框一致性诊断**：两列都以预测框间 IoU≥0.5 为一致。"
             "第一列将任一方缺框计为不一致、分母包含全部样本；第二列排除任一方缺框的样本，"
             "按剩余样本重算类别率。它们是候选诊断口径，不是论文 Consistency 的严格复现。")

lines.append("\n## 四、增益拆解\n")
lines.append("| 路径 | 净增正确条数 | 无框后转正确 | 有框错转正确 | 原正确转错误 |")
lines.append("|---|---:|---:|---:|---:|")
for k, d in decomp.items():
    lines.append(f"| {d['name']}（{k}） | {d['gain']:+d} | {d['fixed']} | "
                 f"{d['improved']} | {d['lost']} |")

lines.append("\n## 五、配对完整性\n")
lines.append("`question_id` 单独使用会重复；`(dataset, image_id)` 也会重复，因为同一张图可有多个问题。")
lines.append("本次使用 `(dataset, image_id, question_id)` 作为复合键，并核验 implicit/explicit 键集合一致。")
lines.append("四组各保留 873 条配对记录，复合键在每组内唯一。")

lines.append("\n## 六、口径声明\n")
lines.append("1. 六类等权平均为论文口径；样本加权作为补充。")
lines.append("2. 无框 = 该条判错，计入分母（严格）。")
lines.append("3. 框提取优先 `<answer>` 标签内，回退全文第一个方括号。")
lines.append("4. 正则支持负号（官方正则会漏掉 `-0.0` 形式）。")
lines.append("5. g3/g4 为自定义 few-shot 探索，非论文协议，不纳入严格复现结论。")
lines.append("6. 配对使用 `(dataset, image_id, question_id)`，并验证 implicit/explicit 查询和 GT 对齐。")

rp = os.path.join(OUT, "复算总表.md")
with open(rp, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n报告已写入 {rp}")

# 机读汇总
out = dict(
    implicit={g: dict(label=LABELS[g], paper=PAPER[g],
                      macro=round(imp[g]["stat"]["macro"], 6),
                      weighted=round(imp[g]["stat"]["weighted"], 6),
                      n=imp[g]["stat"]["n"],
                      n_nobox=imp[g]["stat"]["n_nobox"],
                      nobox_rate=round(imp[g]["stat"]["nobox_rate"], 6),
                      per_class=imp[g]["stat"]["per_class"])
              for g in imp},
    hard={g: {k: (round(v, 6) if isinstance(v, float) else v)
              for k, v in h.items()} for g, h in hard.items()},
    decomp=decomp,
)
jp = os.path.join(OUT, "summary_all.json")
with open(jp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"机读汇总 {jp}")

# 逐条判定
for g in imp:
    with open(os.path.join(OUT, f"per_sample_{g}.jsonl"), "w",
              encoding="utf-8") as f:
        for p in imp[g]["per"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("逐条判定已导出")
