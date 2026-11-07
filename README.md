# DroneVG-R1 / DVGBench 推理复现

基于公开的 Qwen2.5-VL-7B-Instruct 与 DroneVG-R1 adapter，记录六组 implicit 查询实验、四组 explicit 查询补充实验，以及小样本流程和 Swift 模板对照。

**本项目是推理与评测复现，没有重新进行 GRPO 训练。** 四个主要设置的结果趋势与论文一致，但不能称为严格复现；g3/g4 使用自定义文本示例，配对框一致性也仅是诊断指标。

## 从哪里开始

- [统一复算总表](results_full/recompute/复算总表.md)：当前数值入口。
- [复现报告](复现报告.md)：实验设置、主结果、Swift 对照与 explicit 配对说明。
- [目录说明](目录说明.md)：文件与实验的对应关系。
- [第三方来源](SOURCES.md)：论文、数据、基座模型和 adapter。

## 实验与结果

下表准确率均为六类等权宏平均。四组 explicit 实验复用对应 implicit 预测，按 `(dataset, image_id, question_id)` 配对，每组 873 条。

| 组 | 权重 / 提示 | implicit 样本数 | implicit 准确率 | explicit 准确率 | 两查询均正确 |
|---|---|---:|---:|---:|---:|
| g1 | 基座 / CoT | 873 | 42.32% | 57.69% | 34.19% |
| g2 | 基座 / I2E-CoT | 873 | 47.14% | 59.40% | 40.56% |
| g3 | 基座 / CoT + 2 条文本示例 | 870 | 49.29% | — | — |
| g4 | 基座 / CoT + 5 条文本示例 | 867 | 50.24% | — | — |
| g5 | 基座 + adapter / CoT | 873 | 49.23% | 58.82% | 41.51% |
| g6 | 基座 + adapter / I2E-CoT | 873 | 50.63% | 60.03% | 43.07% |

共 **10 组全量推理设置**（6 implicit + 4 explicit），另有 smoke、60/200 条审计和 50 条 explicit 试跑，不重复计入全量组数。

g3/g4 从测试集抽取文本问题与 GT 坐标作为示例，不提供示例图片，并排除示例图片上的全部问题。它们是文本/格式示范探索，不代表论文 few-shot 协议，不能由此断言增益全部来自格式约束。

Swift 200 条配对诊断中，180 条回答逐字相同；两侧均有 91 条正确、33 条无框，没有正确性翻转。该子集未支持“模板导致主结果差距”的归因。

## 原始数据在哪里

| 路径 | 内容 |
|---|---|
| `results_full/ans_g*.jsonl` | 六组 implicit 原始回答 |
| `results_full/dvgb_hard_full/ans_*_explicit_873.jsonl` | 四组 explicit 原始回答 |
| `results_full/dvgb_hard_full/paired_*.jsonl` | 配对明细 |
| `results_full/recompute/` | 统一复算、逐条判定、汇总 JSON |
| `results_full/audit60/`、`audit200/` | 流程与模板诊断 |
| `results_full/dvgb_hard_pilot/` | explicit 试跑和题单 |
| `results_full/server_snapshot/` | 保留的服务器原始结果快照，含重复结果及日志 |
| `audit_reference/dvg_test.jsonl` | 873 条测试标注，含 implicit / explicit 查询 |

原始回答、日志和服务器快照均保留。历史报告可能使用旧口径，引用时以统一复算总表及本 README 的边界说明为准。

## 本地复算（无需 GPU）

安装 Python 3.10 或更新版本，在仓库目录运行：

```shell
python -X utf8 recompute_all.py
python -X utf8 eval_explicit.py
```

两项仅使用 Python 标准库，读取随仓库提供的结果；会更新 `results_full/recompute/` 中相应报告和 JSON，不会重新运行模型。脚本以自身位置定位项目，支持任意克隆路径。

统一复算以预测框与 GT 的 IoU ≥ 0.5 判正确；无可解析框计错。配对诊断的“预测框间 IoU ≥ 0.5”分别报告缺框计失败和排除缺框两种分母。它们**不是论文 Consistency 的严格复现**，也不能单独证明理解能力或解释与论文的分数差距。

## 重新运行推理

`run_exp.py` 提供六组设置；需要单独准备 GPU、PyTorch、Transformers、PEFT、Accelerate、Pillow、模型权重和数据集图片。下载地址见 [SOURCES.md](SOURCES.md)。以下为路径示例，先运行少量样本：

```shell
python run_exp.py --setting g1 --base /path/to/Qwen2.5-VL-7B-Instruct --image-root /path/to/DVGBench/images --questions audit_reference/dvg_test.jsonl --num-samples 30 --out trial_g1.jsonl
```

g5/g6 另传 `--adapter /path/to/DroneVG-R1-7B`。explicit 查询使用 `results_full/dvgb_hard_pilot/questions_explicit_full.jsonl`。为保护归档结果，请使用新的输出路径。

`auto_pipeline.py` 是原 AutoDL 环境的流程脚本，使用 `/root/autodl-tmp` 布局，包含依赖安装和下载步骤，不是跨平台一键重现实验的保证。仓库未锁定全部依赖和模型 revision，重跑的数值可能受版本差异影响。模型权重、图片及完整运行环境未包含在此仓库中。

## 公开归档范围

上传代码、测试标注、实验输出、日志和报告；本地论文全文/PDF、论文截图、依赖 wheel 与 arXiv 缓存不上传，来源链接保留在 `SOURCES.md`。这些本地文件仍保留在原文件夹中。第三方材料遵循各自来源条款，本仓库不对第三方内容重新授权。
