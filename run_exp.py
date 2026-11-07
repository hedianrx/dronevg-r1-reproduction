#!/usr/bin/env python3
"""
DroneVG-R1 复现：6 组对照实验统一推理脚本

组别定义（论文 Figure 11 + Table 7）：
  g1  base + cot      → 43.78   基线
  g2  base + i2e      → 47.93   提示词零样本增益
  g3  base + cot + 2shot → 34.82  few-shot
  g4  base + cot + 5shot → 31.80  few-shot
  g5  droner1 + cot   → 51.24   训练增益锚点（仅正文提及，未单列）
  g6  droner1 + i2e   → 52.05   主结果

关键设计：adapter 直接 PEFT 挂载，不合并
  → 避免"基座 16G + 合并权重 16G"的 32GB 峰值，50GB 数据盘才装得下

用法：
  python run_exp.py --setting g1 --out ans_g1.jsonl
  python run_exp.py --setting g5 --out ans_g5.jsonl --adapter /root/autodl-tmp/DroneVG-R1-7B

  # 先抽样验证（务必先跑这一步）
  python run_exp.py --setting g1 --num-samples 30 --out try.jsonl
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------- 提示词模板（论文 Figure 5 原文，逐字保留双空格） ----------------
PROMPT_COT = (
    "Find <|object_ref_start|>{q}<|object_ref_end|> in the image. Output the thinking process in "
    "<think> </think> and final answer [x1,y1,x2,y2] in <answer> </answer> tags."
)

PROMPT_I2E = (
    "Find <|object_ref_start|>{q}<|object_ref_end|> in the image.  Output the thinking process in "
    "<think> </think>,  a brief description of the referred object (use words "
    "such as color, size, and relative position)  in <explicit> </explicit>, and "
    "the final answer [x1,y1,x2,y2] in <answer> </answer> tags."
)

SHOT_HEADER = "\n\nBelow are {n} examples of solving similar tasks:\n\n{examples}\n\n"
SHOT_FOOTER = "Now solve the task for the new image, following the same format.\n\n"

# ---------------- 6 组配置 ----------------
SETTINGS = {
    "g1": dict(weights="base",    prompt="cot", shots=0, paper="43.78"),
    "g2": dict(weights="base",    prompt="i2e", shots=0, paper="47.93"),
    "g3": dict(weights="base",    prompt="cot", shots=2, paper="34.82"),
    "g4": dict(weights="base",    prompt="cot", shots=5, paper="31.80"),
    "g5": dict(weights="droner1", prompt="cot", shots=0, paper="51.24"),
    "g6": dict(weights="droner1", prompt="i2e", shots=0, paper="52.05"),
}

PROMPTS = {"cot": PROMPT_COT, "i2e": PROMPT_I2E}


def build_fewshot(rows, shots, seed=42):
    """
    随机抽样构造 few-shot 示例。

    ⚠️ 论文未说明示例选择方式，这里用测试集随机抽样并记录，报告中必须声明。
    示例统一用普通 CoT 格式（与 g1 基线一致）。
    """
    rng = random.Random(seed)
    pool = rng.sample(rows, shots)
    blocks = []
    for i, r in enumerate(pool, 1):
        blocks.append(
            f"Example {i}:\n"
            f"Query: {r['question']}\n"
            f"<think>The reference is implicit; I need to locate the object "
            f"matching that description.</think>\n"
            f"<answer>[{r['bbox'][0]:.1f},{r['bbox'][1]:.1f},"
            f"{r['bbox'][2]:.1f},{r['bbox'][3]:.1f}]</answer>"
        )
    return pool, "\n\n".join(blocks)


def resolve_image(root, dataset, image_id):
    for cand in [
        os.path.join(root, dataset, image_id),
        os.path.join(root, image_id),
        os.path.join(root, dataset, "all", image_id),
    ]:
        if os.path.exists(cand):
            return cand
    return None


def load_model(base_path, adapter_path, attn, max_pixels):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if attn == "flash_attention_2":
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_path, **kwargs)

    if adapter_path:
        from peft import PeftModel
        # 关键：不 merge，直接挂载。省 16GB 磁盘，显存占用几乎不变
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[INFO] adapter 已挂载（未合并）: {adapter_path}", flush=True)

    model.eval()
    processor = AutoProcessor.from_pretrained(base_path, use_fast=False)
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=list(SETTINGS))
    ap.add_argument("--base", default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="/root/autodl-tmp/DroneVG-R1-7B",
                    help="仅 g5/g6 使用")
    ap.add_argument("--image-root", default="/root/autodl-tmp/DVGBench/images")
    ap.add_argument("--questions", default="/root/autodl-tmp/DVGBench/dvg_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=0, help="0=全量")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="官方固定 512")
    ap.add_argument("--attn", default="auto",
                    choices=["auto", "sdpa", "flash_attention_2", "eager"],
                    help="auto=ffa 可用则用 flash_attention_2，否则 sdpa")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="覆盖模型默认的最大图像像素数；默认使用模型预处理配置")
    args = ap.parse_args()

    cfg = SETTINGS[args.setting]
    needs_adapter = cfg["weights"] == "droner1"
    if needs_adapter and not os.path.isdir(args.adapter):
        sys.exit(f"[ERROR] {args.setting} 需要 adapter，但不存在: {args.adapter}")

    rows = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    total = len(rows)

    # ---------- few-shot ----------
    shot_prefix = ""
    if cfg["shots"] > 0:
        pool, block = build_fewshot(rows, cfg["shots"])
        exclude = {(r["dataset"], r["image_id"]) for r in pool}
        rows = [r for r in rows if (r["dataset"], r["image_id"]) not in exclude]
        shot_prefix = SHOT_HEADER.format(n=cfg["shots"], examples=block) + SHOT_FOOTER
        meta = dict(setting=args.setting, shots=cfg["shots"],
                    seed=42, note="示例由测试集随机抽样，已从评测集排除",
                    examples=[dict(image_id=r["image_id"], dataset=r["dataset"],
                                   question=r["question"], bbox=r["bbox"])
                              for r in pool])
        with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] {cfg['shots']}-shot 示例已抽取，排除后评测集 {len(rows)} 条"
              f"（原 {total}）")
        print(f"[INFO] 示例记录: {args.out}.meta.json", flush=True)

    if args.num_samples and args.num_samples < len(rows):
        random.seed(42)
        rows = random.sample(rows, args.num_samples)
        print(f"[INFO] 抽样 {len(rows)} 条", flush=True)

    # ---------- 加载 ----------
    attn = args.attn
    if attn == "auto":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
        print(f"[INFO] attn=auto → {attn}", flush=True)
    t0 = time.time()
    model, processor = load_model(
        args.base, args.adapter if needs_adapter else None,
        attn, args.max_pixels)
    print(f"[INFO] 权重={cfg['weights']} prompt={cfg['prompt']} shots={cfg['shots']}")
    print(f"[INFO] 论文参考值 {cfg['paper']}% | 加载耗时 {time.time()-t0:.1f}s",
          flush=True)

    fout = open(args.out, "w", encoding="utf-8")
    n_ok = n_miss = n_fmt = 0
    t_start = time.time()

    for i, r in enumerate(rows):
        path = resolve_image(args.image_root, r["dataset"], r["image_id"])
        if path is None:
            n_miss += 1
            continue

        image = Image.open(path).convert("RGB")
        body = PROMPTS[cfg["prompt"]].format(q=r["question"])
        if shot_prefix:
            body = shot_prefix + body

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": body}
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt").to(model.device)

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None)
        resp = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

        if "<answer>" not in resp or "</answer>" not in resp:
            n_fmt += 1
        n_ok += 1

        fout.write(json.dumps({
            "question_id": r["question_id"], "image_id": r["image_id"],
            "dataset": r["dataset"], "class": r["class"],
            "question": r["question"], "bbox": r["bbox"],
            "answer": resp, "setting": args.setting,
        }, ensure_ascii=False) + "\n")
        fout.flush()

        if (i + 1) % 25 == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(rows) - i - 1) / 60
            print(f"  [{i+1}/{len(rows)}] {el/60:.1f}min "
                  f"均{el/(i+1):.2f}s/条 剩余约{eta:.0f}min", flush=True)

    fout.close()
    el = time.time() - t_start
    print(f"\n[完成] {args.setting}: {n_ok} 条 | 缺图 {n_miss} | "
          f"无answer标签 {n_fmt} | {el/60:.1f}min")
    print(f"[下一步] python compute_metric_fixed.py "
          f"--image-folder {args.image_root} --answers-file {args.out}")


if __name__ == "__main__":
    sys.exit(main())
