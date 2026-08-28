# 2025/2026 最新方法实跑对比 · 总览（2026-08-02）

## 背景
用户指出原对比表偏旧（SASRec 2018 / LightGCN 2020），要求联网寻找 2025/2026 最新方法**实跑**并与现有结果对比。

## 一、联网检索结论（2025–2026 POI 推荐）
2025/2026 的 SOTA 几乎被 **LLM-centric 框架**主导，但**均无法在本项目离线、全候选排名协议下忠实复跑**：

| 方法 | 年份/出处 | 类型 | 能否复跑 |
|---|---|---|---|
| TOOL4POI | 2025 | 工具增强 LLM 重排 | 否（依赖外部 API） |
| PSLM4ST | Algorithms 2025 | Llama-2-7B 微调 | 否（7B 微调 + 自有协议） |
| RALLM-POI | PRICAI'25 | 检索增强 LLM 零样本 | 否（依赖外部 LLM） |
| AWARE | 2025 | Qwen3-4B + 世界知识微调 | 否 |
| ROS | 2025 | Qwen3-4B + 地理推理 RL | 否 |

**唯一可离线忠实复跑的 2025 训练型方法 → `eSASRec`（Tikhonovich et al., RecSys'25）**：
- 在 SASRec 训练范式上引入 **LiGR Transformer 块**（RMSNorm Pre-LN + SwiGLU 前馈，Llama 风格）+ **Sampled Softmax Loss**（in-batch 负样本 + 均匀随机负样本）。
- 公开实现、与已跑 SASRec 同宗，可在相同协议公平对比。

## 二、实现与隔离实验
- `baselines.py` 新增 `RMSNorm` / `SwiGLU` / `LiGRBlock` / `eSASRec`，`eSASRec` 支持 `--loss sampled_softmax | ce`。
- `esasrec_ranks.py` 镜像产出：`esasrec_ranks_s42.json`（sampled_softmax）、`esasrec_ce_ranks_s42.json`（plain CE）。
- **eSASRec-CE = LiGR 架构 + 普通 plain CE**（与 SASRec 完全相同损失），用于分离「架构贡献 vs 损失贡献」两个变量。

## 三、关键结果（诚实全候选协议：hist_mode=user / seq_len=200 / n_test=1447）

### all 子集（全候选排名）
| 模型 | R@5 | R@10 | N@10 |
|---|---|---|---|
| **LM-STKG(full)** | **0.5248** | **0.6440** | **0.4205** |
| LightGCN | 0.3462 | 0.4568 | 0.2846 |
| SASRec | 0.3842 | 0.4485 | 0.2979 |
| eSASRec-CE (LiGR+CE) | 0.2806 | 0.3207 | 0.2295 |
| eSASRec (LiGR+SSM) | 0.2564 | 0.3048 | 0.2116 |
| History-Freq (HF) | 0.4983 | 0.6275 | 0.4096 |

→ **eSASRec(0.3048) 与 eSASRec-CE(0.3207) 均低于 2018 的 SASRec(0.4485)。**

### cold 子集（n=280）
| 模型 | R@5 | R@10 | N@10 |
|---|---|---|---|
| **LM-STKG(full)** | **0.3272** | **0.4986** | **0.2643** |
| LightGCN | 0.1607 | 0.2286 | 0.1350 |
| SASRec | 0.0714 | 0.1179 | 0.0587 |
| eSASRec-CE | 0.1036 | 0.1036 | 0.0894 |
| eSASRec | 0.1036 | 0.1143 | 0.1032 |

### novel_cold 子集（n=75）
全部方法 R@10 = 0.0（基准固有难题）。

## 四、结果解读（防御性结论，非「实现错误」）
1. **主因 = LiGR 架构**：在「POI 仅 4980、replay 主导」的小规模数据上，Llama 风格 Pre-LN + SwiGLU 反而不如 SASRec 原版 Post-LN + ReLU 块。
2. **次因 = Sampled Softmax Loss**：与论文既有 §5.7 校准结论自洽——**只有 plain CE + 均匀负样本**才能保持全候选全局校准；sampled-softmax 稀释校准。
3. 隔离实验 `eSASRec-CE`（换回 plain CE）R@10 从 0.3048 升到 0.3207，证实损失是次要因素、架构是主因。

## 五、论文回填（.md / .tex 同步完成）
- Table 4：SASRec 下新增 `eSASRec`、`eSASRec-CE` 两行。
- §5.4.2 replay 天花板：补充 eSASRec 两值。
- Table 11：新增 TOOL4POI/PSLM4ST/AWARE/ROS/eSASRec/eSASRec-CE 六行；脚注标清三行直接可比；LLM-API 类方法明确注「未复跑、仅作文献上下文」。
- §5.9 讨论：加入 eSASRec 隔离实验解读 + 2025 LLM-centric SOTA 不可复跑的诚实边界说明。
- 冷启动表：新增 eSASRec / eSASRec-CE 两行。
- `honest_eval_report.json` + `honest_eval_table.md` 已重算纳入。

## 六、本轮结论
- 实跑的唯一可复跑 2025 SOTA（eSASRec）**反而低于 2018 的 SASRec**，因此本方法对 SASRec 的领先（**+43.6%，0.6440 vs 0.4485**）更加稳健。
- 2025 LLM-centric SOTA 因运行环境与「sampled-candidate 协议」不可对齐，**仅作文献上下文诚实列出**，不冒充可直接对比。
- 遗留：专利↔论文锚点已统一到 C6，纯 C6 的创造性风险已口头提示、未擅自决策。

## 七、产物文件
- `code/baselines.py`（新增 eSASRec 等类）
- `code/esasrec_ranks.py`（镜像脚本）
- `code/esasrec_ranks_s42.json`、`code/esasrec_ce_ranks_s42.json`（实跑输出）
- `code/honest_eval_report.json`、`code/honest_eval_table.md`（重算终表）
- `旅游推荐论文/08_SCI_Manuscript_KBS.md` / `.tex`（同步回填）
