#!/usr/bin/env python3
"""
DroneVG-R1 评测脚本（论文口径修正版）

相比官方 compute_metric.py 的三处修正：
  1. 【关键】AVG 改为六类等权平均 —— 已用论文表2的9个模型反推验证，误差 ±0.01
  2. 【增强】正则支持负数 —— 官方 [0-9., ]+ 无法匹配模型输出中的负坐标；GT 本身是数值数组
  3. 【增强】框标签优先 —— 优先取 <answer> 内的框，同时保留官方"取第一个"的结果做对照

用法：
  python compute_metric_fixed.py \
      --image-folder ./DVGBench/images \
      --answers-file answer_full.jsonl

输入 jsonl 需含字段：answer, bbox, class, dataset, image_id
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# 官方正则（保留，用于复现论文原始行为）
PAT_OFFICIAL = r"\[([0-9., ]+)\]"
# 修正正则：支持负号
PAT_FIXED = r"\[(-?[0-9.]+(?:\s*,\s*-?[0-9.]+){3})\]"

# 论文表2/表3 使用的六类（顺序即论文列顺序）
CATEGORIES = ["Security", "Traffic", "Social Activity", "Disaster",
              "Productive Activity", "Sport"]

# 测试集 class 字段是简写，需映射到论文类别名
CLASS_MAP = {
    "security": "Security",
    "traffic": "Traffic",
    "social": "Social Activity",
    "social activity": "Social Activity",
    "disaster": "Disaster",
    "productive": "Productive Activity",
    "productive activity": "Productive Activity",
    "sport": "Sport",
}


def parse_boxes(text, pattern):
    return [list(map(float, m.split(","))) for m in re.findall(pattern, text)]


def iou(a, b):
    if len(a) != 4 or len(b) != 4:
        return 0.0
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = xi2 - xi1, yi2 - yi1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pick_box(response, pattern, prefer_answer_tag=True):
    """返回 (box, source)。source ∈ {'answer_tag','first'}"
    """
    if prefer_answer_tag:
        m = re.search(r"<answer>(.*?)</answer>", response, re.S)
        if m:
            inner = parse_boxes(m.group(1), pattern)
            if inner:
                return inner[0], "answer_tag"
    boxes = parse_boxes(response, pattern)
    if boxes:
        return boxes[0], "first"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--dump", default=None, help="可选：导出逐条判定为 json")
    ap.add_argument("--json-out", default=None, help="可选：输出机读汇总 json")
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(args.answers_file, encoding="utf-8")]
    print(f"[INFO] 样本数 {len(preds)}")

    # 三套口径同时统计，便于定位差异
    stats = {
        "official": defaultdict(lambda: [0, 0]),   # 官方：首个框 + 样本加权
        "fixed_tag": defaultdict(lambda: [0, 0]),  # 修正：answer标签框 + 分类计
    }
    overall_off = [0, 0]

    img_size_cache = {}
    dump_rows = []

    for p in tqdm(preds):
        cls_raw = str(p.get("class", "")).strip().lower()
        cat = CLASS_MAP.get(cls_raw, cls_raw)
        gt = p["bbox"]
        resp = p["answer"]

        key = (p.get("dataset"), p.get("image_id"))
        if key not in img_size_cache:
            path = os.path.join(args.image_folder, str(p.get("dataset")), str(p.get("image_id")))
            try:
                w, h = Image.open(path).size
            except Exception:
                w = h = -1
            img_size_cache[key] = (w, h)

        # 官方口径
        box_off, src_off = pick_box(resp, PAT_OFFICIAL, prefer_answer_tag=False)
        ok_off = box_off is not None and iou(gt, box_off) >= 0.5

        # 修正口径
        box_fix, src_fix = pick_box(resp, PAT_FIXED, prefer_answer_tag=True)
        ok_fix = box_fix is not None and iou(gt, box_fix) >= 0.5

        overall_off[0] += int(ok_off)
        overall_off[1] += 1
        stats["official"][cat][0] += int(ok_off)
        stats["official"][cat][1] += 1
        stats["fixed_tag"][cat][0] += int(ok_fix)
        stats["fixed_tag"][cat][1] += 1

        dump_rows.append(dict(qid=p.get("question_id"), image_id=p.get("image_id"),
                              cat=cat, ok_official=ok_off, ok_fixed=ok_fix,
                              src_official=src_off, src_fixed=src_fix,
                              box_official=box_off, box_fixed=box_fix, gt=gt))

    def report(name, table, note):
        print("\n" + "=" * 66)
        print(f"{name}  ({note})")
        print("=" * 66)
        print(f"{'类别':<22}{'样本数':>7}{'正确':>7}{'Acc@0.5':>10}")
        print("-" * 66)
        vals = []
        rates = []
        tot_n = tot_c = 0
        for cat in CATEGORIES:
            c, n = table.get(cat, [0, 0])
            acc = c / n if n else 0.0
            vals.append(acc)
            rates.append(acc)
            tot_n += n
            tot_c += c
            print(f"{cat:<22}{n:>7}{c:>7}{acc*100:>9.2f}%")
        # 不在六类里的样本（若有，提示）
        extra = {k: v for k, v in table.items() if k not in CATEGORIES}
        if extra:
            print(f"\n⚠️ 未归入六类的取值: { {k: v[1] for k, v in extra.items()} }")
        print("-" * 66)
        uni = sum(vals) / len(CATEGORIES) * 100
        wt = (tot_c / tot_n * 100) if tot_n else 0.0
        print(f"{'六类等权平均 (论文口径)':<22}{tot_n:>7}{tot_c:>7}{uni:>9.2f}%")
        print(f"{'样本加权平均 (官方口径)':<22}{tot_n:>7}{tot_c:>7}{wt:>9.2f}%")
        if abs(uni - wt) >= 0.1:
            print(f"  ⚠️ 两种口径相差 {uni-wt:+.2f} 个点，写报告时必须说明用哪个")
        return uni, wt

    uni_fix, wt_fix = report("修正口径（answer标签框 + 负数正则 + 六类等权）",
                             stats["fixed_tag"], "推荐汇报用")
    uni_off, wt_off = report("原始解析规则（首个框 + 原正则）",
                             stats["official"], "对照")
    acc_weighted_official = (overall_off[0] / overall_off[1]
                             if overall_off[1] else 0.0)
    print("\n样本加权（原始解析规则）: "
          f"{acc_weighted_official*100:.2f}%")

    print("\n" + "=" * 66)
    print("论文参考值对照：")
    PAPER = {"g1": 43.78, "g2": 47.93, "g3": 34.82,
             "g4": 31.80, "g5": 51.24, "g6": 52.05}
    setting = preds[0].get("setting") if preds else None
    if setting in PAPER:
        ref = PAPER[setting]
        print(f"  {setting} 论文值 {ref}%  本次 {uni_fix:.2f}%  "
              f"差 {uni_fix-ref:+.2f}")
    print(f"  六类等权(修正解析) {uni_fix:.2f}%  |  "
          f"六类等权(原始解析) {uni_off:.2f}%  |  "
          f"样本加权(修正解析) {wt_fix:.2f}%  |  "
          f"样本加权(原始解析) {acc_weighted_official*100:.2f}%")
    print("=" * 66)

    # 机读汇总，供 auto_pipeline.py 汇总
    if args.json_out:
        summary = dict(
            setting=setting, n=len(preds),
            # Keep machine-readable rates in [0, 1]; human output is percent.
            acc_uniform=round(uni_fix / 100, 4),
            acc_weighted=round(wt_fix / 100, 4),
            acc_uniform_official=round(uni_off / 100, 4),
            acc_weighted_official=round(acc_weighted_official, 4),
            paper=PAPER.get(setting),
            per_class={c: round(stats["fixed_tag"].get(c, [0, 0])[0] /
                                stats["fixed_tag"].get(c, [0, 1])[1] * 100, 2)
                       if stats["fixed_tag"].get(c, [0, 0])[1] else None
                       for c in CATEGORIES},
            n_per_class={c: stats["fixed_tag"].get(c, [0, 0])[1]
                         for c in CATEGORIES},
        )
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"机读汇总已写入: {args.json_out}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for r in dump_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n逐条判定已导出: {args.dump}")


if __name__ == "__main__":
    main()
