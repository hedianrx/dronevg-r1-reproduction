# 数据与第三方来源

本仓库为独立的推理复现归档，不是作者官方仓库，也不包含重新训练的模型。

| 材料 | 来源 | 本仓库用途 |
|---|---|---|
| DVGBench 论文 | https://arxiv.org/abs/2601.00998 | 实验设计和结果对照；全文及截图留在本地 |
| 官方代码 | https://github.com/VisionXLab/DVGBench | `audit_reference/compute_metric.py`、`inference_rec_qwen2_5*.py` 和 `github_README.md` 对照副本 |
| DVGBench 数据 | https://huggingface.co/datasets/erenzhou/DVGBench | `audit_reference/dvg_test.jsonl`、数据卡及派生题单；不含图片 |
| 基座模型 | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | 推理基座及预处理配置；不含权重 |
| DroneVG-R1 adapter | https://huggingface.co/datasets/erenzhou/DroneVG-R1-7B | g5/g6 使用的公开 adapter；不含权重 |
| ms-swift | https://github.com/modelscope/ms-swift | `audit_reference/swift_qwen.py` 参考副本及模板诊断 |
| qwen-vl-utils | https://pypi.org/project/qwen-vl-utils/ | 本地保存的 0.0.14 wheel 不上传 |

保存的数据卡声明数据集为 Apache-2.0，详见 `audit_reference/README.md`；该声明不代表论文、所有第三方代码、原始图片或模型均使用同一许可证。第三方文件保留各自来源条款，使用前请核对上游许可。没有给整个混合归档套用统一许可证。

参考副本用于审计原实验，未完整记录所有上游 commit 或模型 revision；不能据此承诺从任意最新版本获得逐字相同的推理结果。

论文标题：*DVGBench: Implicit-to-Explicit Visual Grounding Benchmark in UAV Imagery with Large Vision-Language Models*。作者信息及引用条目保存在数据卡和上游仓库中。
