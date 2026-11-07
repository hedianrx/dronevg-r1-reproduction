#!/usr/bin/env python3
"""
DroneVG-R1 复现 · 总控流水线
=============================

一条命令跑完：环境自检 → 依赖安装 → 资源下载 → 解压 →
冒烟测试 → 6 组推理 → 算分 → 汇总报告

设计原则：
  * 幂等 —— 已有资源和完整实验组会跳过；未完成实验组从头重跑
  * 自愈 —— 单组失败不影响后续组，最后统一报告
  * 可观测 —— 全程写日志到 logs/，另有一份人读的 PROGRESS.log

用法（在 Aut oDL 机器上，/root/autodl-tmp 下）：
    python auto_pipeline.py                 # 全流程
    python auto_pipeline.py --from smoke    # 从冒烟测试开始（已装好环境时）
    python auto_pipeline.py --only g1,g5    # 只跑指定组
    python auto_pipeline.py --skip-download # 跳过下载（资源已在）
    python auto_pipeline.py --smoke-n 30    # 冒烟样本数

⚠️ 长时间运行必须用 nohup：
    nohup python auto_pipeline.py > logs/pipeline.out 2>&1 &
    tail -f logs/pipeline.out
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ============================ 路径与常量 ============================

WORK = "/root/autodl-tmp"

PATHS = dict(
    base=f"{WORK}/Qwen2.5-VL-7B-Instruct",
    adapter=f"{WORK}/DroneVG-R1-7B",
    data=f"{WORK}/DVGBench",
    images=f"{WORK}/DVGBench/images",
    test_jsonl=f"{WORK}/DVGBench/dvg_test.jsonl",
    scripts=f"{WORK}/scripts",
    logs=f"{WORK}/logs",
    results=f"{WORK}/results",
)

# 6 组：论文参考值来自 Table 2 / Table 7 / Figure 11
SETTINGS = [
    # (key, 权重, 提示词, shot, 论文值, 说明)
    ("g1", "base",    "cot", 0, 43.78, "基座 + 普通 CoT —— 基线"),
    ("g5", "droner1", "cot", 0, 51.24, "DroneVG-R1 + 普通 CoT —— 训练增益锚点"),
    ("g2", "base",    "i2e", 0, 47.93, "基座 + I2E-CoT —— 提示词零样本增益"),
    ("g6", "droner1", "i2e", 0, 52.05, "DroneVG-R1 + I2E-CoT —— 主结果"),
    ("g3", "base",    "cot", 2, 34.82, "基座 + CoT + 2-shot"),
    ("g4", "base",    "cot", 5, 31.80, "基座 + CoT + 5-shot"),
]

MIN_DISK_GB = 25
BASE_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
]
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


# ============================ 工具函数 ============================

class Log:
    """双输出：屏幕 + logs/pipeline_YYYYmmdd_HHMMSS.log"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg="", flush=True):
        line = f"[{datetime.now():%H:%M:%S}] {msg}" if msg else ""
        print(line, flush=flush)
        self.f.write(line + "\n")
        self.f.flush()

    def section(self, title):
        self("")
        self("=" * 68)
        self(f" {title}")
        self("=" * 68)


def run(cmd, log, check=True, capture=False, timeout=None):
    """执行子进程，日志实时透传。返回 (returncode, stdout_text)"""
    log(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    p = subprocess.run(cmd, shell=isinstance(cmd, str),
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None,
                       text=True, timeout=timeout)
    out = p.stdout or ""
    if capture:
        for line in out.rstrip().split("\n")[-30:]:
            log(f"  {line}")
    if check and p.returncode != 0:
        log(f"  ❌ 返回码 {p.returncode}")
        raise RuntimeError(f"命令失败: {cmd}")
    return p.returncode, out


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except Exception:
        return -1


def py_has(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


# ============================ 各阶段 ============================

def stage_env(log, skip_install=False):
    """阶段 0：环境自检 + 依赖安装"""
    log.section("阶段 0 / 环境自检")

    # GPU
    rc, out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"], log, check=False, capture=True)
    gpu_ok = (rc == 0)
    if gpu_ok:
        log(f"  GPU: {out.strip()}")
    else:
        log("  ⚠️ 无 GPU 可见 —— 若在无卡模式下属正常；"
            "推理阶段必须有 GPU，请切回有卡模式后再跑")

    # Python / torch
    import torch
    log(f"  Python {sys.version.split()[0]} | torch {torch.__version__} "
        f"| cuda {torch.version.cuda} | avail {torch.cuda.is_available()}")

    # 磁盘
    free = disk_free_gb(WORK)
    log(f"  磁盘可用 {free:.1f}GB（需 ≥{MIN_DISK_GB}GB）")
    if free < MIN_DISK_GB:
        raise RuntimeError(f"磁盘不足：{free:.1f}GB < {MIN_DISK_GB}GB")

    if skip_install:
        log("  --skip-install，跳过依赖安装")
        return gpu_ok

    # HF 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    log("  HF_ENDPOINT=https://hf-mirror.com")
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        if "HF_ENDPOINT" not in open(bashrc).read():
            with open(bashrc, "a") as f:
                f.write('\nexport HF_ENDPOINT=https://hf-mirror.com\n')
    except Exception:
        pass

    log("\n  安装依赖（已装则跳过，约 1~3 分钟）...")
    run(f"{sys.executable} -m pip install -q -U pip", log, check=False)
    run(f"{sys.executable} -m pip install -q "
        '"transformers>=4.49,<5" "peft>=0.13" "accelerate>=0.30" '
        "pillow tqdm numpy", log)
    import transformers
    import peft
    log(f"  transformers {transformers.__version__} | peft {peft.__version__}")

    # flash-attn：锁版本，避免退化成源码编译
    if (torch.version.cuda or "").startswith("12"):
        log("\n  flash-attn：尝试预编译 wheel（失败自动走 sdpa，不影响结论）")
        installed = False
        for ver in ("2.7.4.post1", "2.6.3"):
            rc, _ = run(f"{sys.executable} -m pip install -q flash-attn=={ver} "
                        "--no-build-isolation", log, check=False, capture=True,
                        timeout=1800)
            if rc == 0:
                installed = True
                break
            log(f"  {ver} 未装上，换下一个版本")
        if installed:
            try:
                import flash_attn
                log(f"  ✅ flash-attn {flash_attn.__version__}")
            except Exception as e:
                log(f"  ⚠️ 装上但导入失败（{e}），推理会走 sdpa")
        else:
            log("  ⚠️ flash-attn 全部失败，使用 sdpa（慢 10~15%，不影响结论）")
    else:
        log(f"  CUDA {torch.version.cuda}，跳过 flash-attn")

    return gpu_ok


def stage_download(log):
    """阶段 1：下载资源"""
    log.section("阶段 1 / 下载资源（约 20GB，首次最慢）")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if not py_has("huggingface_hub"):
        run(f"{sys.executable} -m pip install -q -U huggingface_hub", log)

    from huggingface_hub import snapshot_download

    # 1.1 基座
    if all(os.path.exists(os.path.join(PATHS["base"], f)) for f in BASE_FILES):
        log("  ✓ 基座已完整，跳过")
    else:
        log("  → 下载 Qwen2.5-VL-7B-Instruct（约 16GB）...")
        snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct",
                          local_dir=PATHS["base"],
                          ignore_patterns=["*.md", "*.pth", "*.gitattributes"])
        log("  ✓ 基座下载完成")

    # 1.2 adapter（注意：在 Datasets 仓库，不是 Models）
    if all(os.path.exists(os.path.join(PATHS["adapter"], f))
           for f in ADAPTER_FILES):
        log("  ✓ adapter 已存在，跳过")
    else:
        log("  → 下载 DroneVG-R1-7B adapter（约 0.3GB）...")
        snapshot_download("erenzhou/DroneVG-R1-7B", repo_type="dataset",
                          local_dir=PATHS["adapter"],
                          allow_patterns=ADAPTER_FILES)
        log("  ✓ adapter 下载完成")

    # 1.3 数据集
    if os.path.exists(PATHS["test_jsonl"]):
        log("  ✓ DVGBench 已存在，跳过")
    else:
        log("  → 下载 DVGBench（图像 zip 约 0.7GB）...")
        snapshot_download("erenzhou/DVGBench", repo_type="dataset",
                          local_dir=PATHS["data"],
                          ignore_patterns=["*.DS_Store", "**/.DS_Store"])
        log("  ✓ DVGBench 下载完成")

    # 1.4 解压图像
    if os.path.isdir(PATHS["images"]) and os.listdir(PATHS["images"]):
        n = len(os.listdir(PATHS["images"]))
        log(f"  ✓ 图像已解压（{n} 个）")
    else:
        zips = [f for f in os.listdir(PATHS["data"]) if f.endswith(".zip")]
        if not zips:
            raise RuntimeError(f"{PATHS['data']} 下找不到 zip")
        for z in zips:
            log(f"  → 解压 {z} ...")
            run(f'cd "{PATHS["data"]}" && unzip -q -o "{z}"', log)
        log(f"  ✓ 解压完成（{len(os.listdir(PATHS['images']))} 个文件）")


def stage_verify(log):
    """阶段 2：文件完整性校验"""
    log.section("阶段 2 / 文件完整性校验")

    checks = [
        ("基座 config",     os.path.join(PATHS["base"], "config.json")),
        *[(f"基座分片 {i}/{len(BASE_FILES) - 2}",
           os.path.join(PATHS["base"], name))
          for i, name in enumerate(BASE_FILES[2:], 1)],
        ("adapter config",  os.path.join(PATHS["adapter"], "adapter_config.json")),
        ("adapter 权重",    os.path.join(PATHS["adapter"], "adapter_model.safetensors")),
        ("测试集",          PATHS["test_jsonl"]),
        ("图像目录",        PATHS["images"]),
    ]
    bad = []
    for name, p in checks:
        ok = os.path.exists(p)
        log(f"  {'✓' if ok else '✗'} {name}")
        if not ok:
            bad.append(name)

    if os.path.exists(PATHS["test_jsonl"]):
        rows = sum(1 for _ in open(PATHS["test_jsonl"], encoding="utf-8"))
        log(f"  测试集条数: {rows}（应为 873）")
        if rows != 873:
            raise RuntimeError("测试集条数不为 873，请确认使用 dvg_test.jsonl")

    # adapter_config 记录的是作者本地路径，明确说明不影响
    acfg = os.path.join(PATHS["adapter"], "adapter_config.json")
    if os.path.exists(acfg):
        cfg = json.load(open(acfg))
        log(f"  adapter 记录的基座: {cfg.get('base_model_name_or_path')}")
        log("    （作者机器路径；run_exp.py 会显式传 --base，不受影响）")

    if bad:
        raise RuntimeError(f"缺失文件: {bad}")

    n_img = sum(len(files) for _, _, files in os.walk(PATHS["images"]))
    log(f"  图像文件总数: {n_img}")
    if n_img < 800:
        raise RuntimeError(f"图像数量异常: {n_img} < 800")

    missing = []
    with open(PATHS["test_jsonl"], encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            dataset, image_id = str(row["dataset"]), str(row["image_id"])
            candidates = [
                os.path.join(PATHS["images"], dataset, image_id),
                os.path.join(PATHS["images"], image_id),
                os.path.join(PATHS["images"], dataset, "all", image_id),
            ]
            if not any(os.path.isfile(path) for path in candidates):
                missing.append((line_no, dataset, image_id))
    if missing:
        preview = ", ".join(f"line {n}: {d}/{i}" for n, d, i in missing[:5])
        raise RuntimeError(f"测试集有 {len(missing)} 条找不到图像，例如 {preview}")
    log("  ✓ 测试集每条记录均能匹配到图像")


def stage_smoke(log, n, results_dir):
    """阶段 3：冒烟测试 —— 30 条，验证能出框、能打分"""
    log.section(f"阶段 3 / 冒烟测试（g1 抽 {n} 条）")

    out = os.path.join(results_dir, "smoke_g1.jsonl")
    run([sys.executable, os.path.join(PATHS["scripts"], "run_exp.py"),
         "--setting", "g1", "--num-samples", str(n), "--out", out], log)

    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    n_ans = sum(1 for r in rows if "</answer>" in r["answer"])
    box_pattern = r"\[(-?[0-9.]+(?:\s*,\s*-?[0-9.]+){3})\]"
    n_box = sum(bool(re.search(box_pattern, r["answer"])) for r in rows)
    log(f"\n  输出 {len(rows)} 条 | 含 </answer> {n_ans} | 可解析坐标框 {n_box}")
    log("  首条样例：")
    log("  " + rows[0]["answer"][:300].replace("\n", " ⏎ "))

    rate = n_ans / max(len(rows), 1)
    box_rate = n_box / max(len(rows), 1)
    if box_rate < 0.5:
        raise RuntimeError(
            f"坐标框解析率异常：answer 标签 {rate*100:.0f}%，"
            f"可解析坐标框 {box_rate*100:.0f}%，低于 50%。"
            "请检查提示词或模型加载。")
    if rate < 0.8 or box_rate < 0.8:
        log("  ⚠️ 格式率低于 80%；将格式失败计入后续实验结果")
    log(f"  ✅ 冒烟通过（answer 标签 {rate*100:.0f}%，"
        f"可解析坐标框 {box_rate*100:.0f}%）")
    return True


def stage_group(log, key, weights, prompt, shots, results_dir, adapter_dir,
                num_samples=0):
    """跑一组推理 + 算分"""
    log.section(f"实验组 {key} / {weights} + {prompt} + {shots}-shot")

    ans_file = os.path.join(results_dir, f"ans_{key}.jsonl")
    sum_file = os.path.join(results_dir, f"summary_{key}.json")
    log_file = os.path.join(results_dir, f"run_{key}.log")

    with open(PATHS["test_jsonl"], encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    if shots:
        pool = random.Random(42).sample(questions, shots)
        excluded = {(row["dataset"], row["image_id"]) for row in pool}
        questions = [row for row in questions
                     if (row["dataset"], row["image_id"]) not in excluded]
    expected = min(num_samples or len(questions), len(questions))
    if os.path.exists(ans_file) and os.path.exists(sum_file):
        try:
            with open(sum_file, encoding="utf-8") as f:
                old_summary = json.load(f)
            with open(ans_file, encoding="utf-8") as f:
                answer_count = sum(1 for line in f if line.strip())
            if (old_summary.get("setting") == key
                    and old_summary.get("n") == expected
                    and answer_count == expected):
                log(f"  ✓ {key} 已有完整结果（{expected} 条），跳过")
                return True
        except (OSError, ValueError):
            pass

    # 推理
    cmd = [sys.executable, os.path.join(PATHS["scripts"], "run_exp.py"),
           "--setting", key, "--out", ans_file, "--adapter", adapter_dir]
    if num_samples:
        cmd += ["--num-samples", str(num_samples)]
    with open(log_file, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    if p.returncode != 0:
        log(f"  ❌ 推理失败（返回码 {p.returncode}），详见 {log_file}")
        tail = open(log_file, encoding="utf-8", errors="replace").read()[-1500:]
        log(tail)
        return False

    with open(ans_file, encoding="utf-8") as f:
        answer_count = sum(1 for line in f if line.strip())
    if answer_count != expected:
        log(f"  ❌ 输出条数 {answer_count}，预期 {expected}；拒绝计算分数")
        return False

    # 抽最后几行进度
    try:
        lines = open(log_file, encoding="utf-8", errors="replace").read()
        for line in [l for l in lines.split("\n") if l.strip()][-4:]:
            log(f"  {line}")
    except Exception:
        pass

    # 算分
    cmd = [sys.executable, os.path.join(PATHS["scripts"],
                                        "compute_metric_fixed.py"),
           "--image-folder", PATHS["images"],
           "--answers-file", ans_file,
           "--json-out", sum_file,
           "--dump", os.path.join(results_dir, f"detail_{key}.jsonl")]
    with open(os.path.join(results_dir, f"metric_{key}.log"), "w",
              encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    if p.returncode != 0:
        log(f"  ❌ 算分失败（返回码 {p.returncode}）")
        log(open(os.path.join(results_dir, f"metric_{key}.log"),
                 encoding="utf-8", errors="replace").read()[-1200:])
        return False

    if os.path.exists(sum_file):
        s = json.load(open(sum_file, encoding="utf-8"))
        ref = s.get("paper")
        delta = f"| 论文 {ref}% 差 {s['acc_uniform']*100-ref:+.2f}" if ref else ""
        log(f"  ✅ {key}: 六类等权 {s['acc_uniform']*100:.2f}%  "
            f"| 样本加权(修正解析) {s['acc_weighted']*100:.2f}%  "
            f"| 样本加权(原始解析) "
            f"{s.get('acc_weighted_official', s['acc_weighted'])*100:.2f}% {delta}")
        log(f"     {s['per_class']}")
        return True
    return False


def stage_report(log, results_dir, report_path):
    """阶段 5：汇总报告"""
    log.section("阶段 5 / 汇总")

    order = [s[0] for s in SETTINGS]
    meta = {s[0]: s for s in SETTINGS}
    data = {}
    for key in order:
        p = os.path.join(results_dir, f"summary_{key}.json")
        if os.path.exists(p):
            data[key] = json.load(open(p, encoding="utf-8"))

    lines = []
    lines.append("# DroneVG-R1 复现结果汇总\n")
    lines.append(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n")
    lines.append("## 六组对照\n")
    lines.append("| 组 | 权重 | 提示词 | shot | 论文 | 本次(六类等权) | 差值 "
                 "| 样本加权(修正解析) | 样本加权(原始解析) | 说明 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in order:
        _, w, pr, sh, ref, note = meta[key]
        if key in data:
            s = data[key]
            got = s["acc_uniform"] * 100
            wt = s["acc_weighted"] * 100
            wt_official = s.get("acc_weighted_official", s["acc_weighted"]) * 100
            lines.append(f"| {key} | {w} | {pr} | {sh} | {ref}% | "
                         f"**{got:.2f}%** | {got-ref:+.2f} | {wt:.2f}% | "
                         f"{wt_official:.2f}% | {note} |")
        else:
            lines.append(f"| {key} | {w} | {pr} | {sh} | {ref}% | 未完成 | — | — | — | {note} |")

    # 增益拆解
    if all(k in data for k in ("g1", "g2", "g5", "g6")):
        a = {k: data[k]["acc_uniform"] * 100 for k in ("g1", "g2", "g5", "g6")}
        lines.append("\n## 增益拆解（本次实测 + 论文值）\n")
        lines.append("```")
        lines.append("实测路径：")
        lines.append(f"  g1 基座+CoT        {a['g1']:.2f}")
        lines.append(f"  g2 基座+I2E-CoT    {a['g2']:.2f}   "
                     f"(提示词增益 {a['g2']-a['g1']:+.2f})")
        lines.append(f"  g5 DroneVG+CoT     {a['g5']:.2f}   "
                     f"(训练增益   {a['g5']-a['g1']:+.2f})")
        lines.append(f"  g6 DroneVG+I2E     {a['g6']:.2f}   "
                     f"(训练后提示词增益 {a['g6']-a['g5']:+.2f})")
        lines.append("")
        lines.append("论文值：")
        lines.append("  43.78 ──提示词──> 47.93 ──训练──> 51.24 ──提示词──> 52.05")
        lines.append("        +4.15          +3.31          +0.81")
        lines.append("")
        lines.append("  另一种拆解（口径不同，同样成立）：")
        lines.append("  43.78 ──训练──> 51.24 (+7.46) ──提示词──> 52.05 (+0.81)")
        lines.append("  43.78 ──提示词──> 47.93 (+4.15) ──训练──> 52.05 (+4.12)")
        lines.append("```")

    # 分类别
    lines.append("\n## 分类别准确率\n")
    lines.append("| 组 | " + " | ".join(
        ["Security", "Traffic", "Social", "Disaster", "Productive", "Sport"]) + " |")
    lines.append("|---|" + "---|" * 6)
    for key in order:
        if key in data:
            pc = data[key]["per_class"]
            row = [f"{pc[c]:.1f}" if pc.get(c) is not None else "—"
                   for c in ["Security", "Traffic", "Social Activity",
                             "Disaster", "Productive Activity", "Sport"]]
            lines.append(f"| {key} | " + " | ".join(row) + " |")

    # 声明
    lines.append("\n## 口径声明（必读）\n")
    lines.append("1. **六类等权平均**按当前复现假设作为论文对照口径；样本加权结果单独列出，"
                 "两者差值取决于本次各类别准确率，不预设固定差值。")
    lines.append("2. **正则修正**：官方 `\\[([0-9., ]+)\\]` 不支持模型输出中的负坐标；"
                 "GT 从 JSON 数值数组读取，不经过此正则。")
    lines.append("3. **框选取**：优先取 `<answer>` 标签内的框，官方是取全文第一个方括号。")
    lines.append("4. **few-shot 示例**为测试集随机抽样（seed=42）且已从评测集排除；"
                 "所以 g3/g4 实际评测数分别为 870/867（同图多题一起排除）；论文未说明示例选取方式，"
                 "few-shot 分数不应视为严格同协议对比。")
    lines.append("5. **图像像素配置**：默认沿用基座预处理配置（min_pixels=3136，"
                 "max_pixels=12845056）；与官方 Swift 的图像缩放细节仍可能有差异。")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  报告已写入: {report_path}")
    log("\n" + "\n".join(lines[:40]))


# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_stage", default="env",
                    choices=["env", "download", "verify", "smoke", "run", "report"])
    ap.add_argument("--only", default="", help="只跑指定组，逗号分隔，如 g1,g5")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-download", action="store_true",
                    help="跳过下载阶段，仍执行文件和图像校验")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=30)
    ap.add_argument("--num-samples", type=int, default=0, help="每组抽样数，0=全量")
    args = ap.parse_args()
    if args.num_samples < 0 or args.smoke_n < 1:
        ap.error("--num-samples 不能为负数，--smoke-n 必须大于 0")

    for d in (PATHS["logs"], PATHS["results"]):
        os.makedirs(d, exist_ok=True)
    log = Log(os.path.join(PATHS["logs"],
                           f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"))

    log(f"启动 {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"工作目录 {WORK} | 脚本目录 {PATHS['scripts']}")

    t0 = time.time()
    stages = ["env", "download", "verify", "smoke", "run", "report"]
    start_idx = stages.index(args.from_stage)

    try:
        if start_idx <= 0:
            gpu_ok = stage_env(log, args.skip_install)
        else:
            # 跳过 env 阶段时也要探一次 GPU，否则断点续跑会裸奔到推理
            import torch
            gpu_ok = torch.cuda.is_available()
            log(f"跳过 env 阶段 | GPU 可用: {gpu_ok}")

        if start_idx <= 1 and not args.skip_download:
            stage_download(log)
        elif start_idx <= 1:
            log("  --skip-download，跳过资源下载")
        if start_idx <= 2:
            stage_verify(log)

        need_gpu = start_idx <= 4   # smoke 和 run 都需要 GPU
        if need_gpu and not gpu_ok:
            raise RuntimeError("推理需要 GPU，当前不可见。请切回有卡模式后重试。")

        if start_idx <= 3 and not args.skip_smoke:
            stage_smoke(log, args.smoke_n, PATHS["results"])

        if start_idx <= 4:
            only = [s.strip() for s in args.only.split(",") if s.strip()]
            todo = [s for s in SETTINGS if not only or s[0] in only]
            log.section(f"阶段 4 / 串行推理 {len(todo)} 组")
            log(f"  顺序: {[s[0] for s in todo]}")
            ok, fail = [], []
            for key, w, pr, sh, ref, note in todo:
                t = time.time()
                done = stage_group(log, key, w, pr, sh, PATHS["results"],
                                   PATHS["adapter"], args.num_samples)
                (ok if done else fail).append(key)
                log(f"  该组耗时 {(time.time()-t)/60:.1f} 分钟")
                log(f"  累计 {(time.time()-t0)/60:.1f} 分钟 | "
                    f"完成 {ok} | 失败 {fail}")
            if fail:
                log(f"\n  ⚠️ 失败组: {fail}（其余组不受影响）")

        report_path = os.path.join(PATHS["results"], "复现报告.md")
        stage_report(log, PATHS["results"], report_path)

        log("")
        log("=" * 68)
        log(f" 全部完成，总耗时 {(time.time()-t0)/60:.1f} 分钟")
        log(f" 报告: {report_path}")
        log(f" 结果目录: {PATHS['results']}")
        log("=" * 68)
        log("")
        log("提醒：确认结果无误后再关机。")
        log("      关机前把 results/ 目录拉回本地，实例释放后数据会丢。")
        return 0

    except Exception as e:
        log("")
        log("!" * 68)
        log(f" ❌ 流水线中断: {e}")
        log("!" * 68)
        log("已完成的阶段会保留。修复后从断点续跑，例如：")
        log("  python auto_pipeline.py --from run --only g3,g4")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
