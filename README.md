# AI Rumor Detection — 可解释谣言检测系统

**2026《人工智能导论》课程大作业 · 第六组**

> 📄 **大作业报告**: [report.pdf](report.pdf)

## 📑 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [训练历程](#训练历程)
- [最终评估](#最终评估)
- [对抗训练](#对抗训练)
- [关键设计决策](#关键设计决策)
- [已知问题](#已知问题)
- [文件结构](#文件结构)
- [团队](#团队)

## 项目概述

构建**可解释谣言检测系统**：输入推文文本 + 事件 ID，输出（1）谣言/非谣言分类，（2）判断依据的自然语言解释。

- **分类模型**: RoBERTa-large (355M)，微调，验证集准确率 **89.53%**
- **解释模型**: DeepSeek-V3.2 via SJTU API，中文解释，五要素 Prompt
- **检索模型**: sentence-transformers (all-MiniLM-L6-v2)，训练集语义相似度检索
- **关键词提取**: RoBERTa 末层多头注意力平均排序，输出 Top-5 判别词
- **置信度分级**: 三级体系（确信 / 倾向 / 存疑），助用户识别不确定场景
- **对抗鲁棒性**: 三模型（干净 / 对抗 V1 / 对抗 V2），翻转率从 6.4% 降至 5.6%

---

## 快速开始

### 环境

```bash
pip install torch transformers sentence-transformers openai pandas numpy scikit-learn matplotlib seaborn tqdm python-dotenv nltk
```

### 运行（一键复现）

**步骤 0：获取代码**

```bash
git clone https://github.com/pnnooy/AI_rumor_detection.git
cd AI_rumor_detection
```

**步骤 1：安装环境**（一次性，约 3 分钟）

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

**步骤 2：下载模型权重**

> 下载链接：[Rumor_detection — SJTU Pan](https://pan.sjtu.edu.cn/web/share/bb890ce0551de5fe239de6b8b1673e88)

将以下文件下载并放入对应目录（检索索引 `data/index.pkl` 已在仓库中，无需下载）：

| 文件 | 放置路径 | 大小 |
|------|------|------|
| 干净训练基线 | `checkpoints/model_clean.pt` | ~1.3 GB |
| 对抗V1（同义词防御） | `checkpoints/model_adv_v1.pt` | ~1.3 GB |
| 对抗V2（多攻击防御） | `checkpoints/model_adv_v2.pt` | ~1.3 GB |

**步骤 3：分类推理**（~4 分钟，纯本地 CPU，不联网）

RoBERTa-large 对 401 条验证集逐条判断谣言/非谣言，输出预测标签 + 置信度 + 关键词。

```bash
python inference.py --input rumer2026/val.csv --output results/val_results.csv --no-llm
```

**步骤 4：LLM 中文解释**（需配置 API Key）

> ⚠️ SJTU API 限速 10 次/分钟，全量 401 条约需 39 分钟，非代码效率问题。可选随机抽取 10 条进行快速测试。

```bash
cp .env.example .env        # 编辑 .env 填入 SJTU_API_KEY

# 快速体验：随机抽取 10 条，约 1 分钟
python inference.py --input rumer2026/val.csv --output results/val_results_full.csv --limit 10

# 全量 401 条，约 39 分钟
python inference.py --input rumer2026/val.csv --output results/val_results_full.csv
```

**步骤 5：三模型鲁棒性对比**（~15 分钟，核心验证，不调 API，不联网）

对三个模型生成同义词对抗样本并对比翻转率：
- `model_clean`  — 正常训练，未见过对抗样本（基线）
- `model_adv_v1` — 训练时注入 WordNet 同义词攻击（单防御）
- `model_adv_v2` — 训练时注入三种随机攻击（综合防御）

```bash
python adversarial.py --mode compare --original rumer2026/val.csv --seed 42
```

**步骤 6：评估 + 出图**（~10 秒）

生成 6 张评估图表（混淆矩阵、事件准确率、置信度分布等）及终端指标表。

```bash
python evaluate.py --input results/val_results.csv --output-dir figures/
```

---

## 系统架构

```
推文文本 + event_id
    │
    ├──→ [RoBERTa-large 分类器] ──→ label (0/1) + confidence + 关键词
    │                                        │
    ├──→ [sentence-transformers 检索] ──→ Top-3 相似训练案例
    │                                        │
    └──→ [DeepSeek-V3.2 LLM] ←────── 五要素 prompt ──┘
              │
              └──→ 中文解释
```

### 技术栈

| 组件 | 技术 | 运行方式 |
|------|------|----------|
| 分类器 | RoBERTa-large (HuggingFace AutoModel) | 本地 CPU/GPU |
| 事件特征 | Event Embedding（7×32 维向量，与 [CLS] 拼接） | 本地 |
| 关键词提取 | Attention Weights（末层所有 head 平均） | 本地 |
| 案例检索 | sentence-transformers all-MiniLM-L6-v2 | 本地离线 |
| LLM 解释 | DeepSeek-V3.2 `deepseek-chat` via SJTU API | 远程 API |
| API 端点 | `https://models.sjtu.edu.cn/api/v1` | 仅校园网/VPN |

### Event Embedding 设计

事件信息通过独立的 Embedding 层注入模型：7 个事件各对应 32 维向量，与文本 [CLS] 表示拼接后送入分类头。相比传统 `[EVENT_N]` 特殊 token 方式，该设计避免事件 token 与文本词 token 在 self-attention 中的不必要交互——event 是全局上下文，不应与单个词做 attention。

### LLM 角色：透明解释者，非独立法官

经过实验验证，LLM 作为"独立法官"尝试纠正 DL 模型会导致准确率从 89% 降至 65%。原因：LLM 没有训练集标签知识，其常识判断与数据集标注标准不一致。最终方案为 LLM 作为 **DL 判断的透明解释者**，而非二审法官。

---

## 训练历程

### 模型演进

| 阶段 | 模型 | 最佳配置 | Val Acc | Val F1 | 提升 |
|------|------|------|:---:|:---:|:---:|
| 基线 | BERT-base (110M) | lr=2e-5, 5ep | 82.79% | 79.88% | — |
| V1 搜索 | BERT-base | lr=3e-5, max_len=128, 12ep | 85.04% | 82.56% | +2.25 |
| V2 搜索 | RoBERTa-base (125M) | lr=3e-5, max_len=128, 12ep | 87.78% | 86.20% | +4.99 |
| **⭐ V2 最佳** | **RoBERTa-large (355M)** | **lr=1e-5, max_len=128, 10ep** | **89.53%** | **87.65%** | **+6.74** |

### V2 完整实验结果（11 组）

训练环境：4×3090 GPU，HF_ENDPOINT=https://hf-mirror.com，统一 seed=42

| 排名 | 编号 | 模型 | epoch | bs | lr | max_len | dropout | val_acc | val_f1 | prec | rec | 耗时 |
|------|------|------|:---:|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 1 | L5 | RoBERTa-large | 10 | 16 | 1e-5 | 128 | 0.2 | **0.8953** | **0.8765** | 0.9030 | 0.8514 | 9.8m |
| 2 | L4 | RoBERTa-large | 10 | 16 | 3e-5 | 128 | 0.2 | 0.8878 | 0.8739 | 0.8571 | 0.8914 | 9.7m |
| 3 | L2 | RoBERTa-large | 8 | 32 | 3e-5 | 64 | 0.3 | 0.8853 | 0.8678 | 0.8728 | 0.8629 | 4.7m |
| 4 | L1 | RoBERTa-large | 8 | 32 | 2e-5 | 64 | 0.3 | 0.8803 | 0.8605 | 0.8757 | 0.8457 | 4.4m |
| 5 | r4 | RoBERTa-base | 12 | 48 | 3e-5 | 128 | 0.2 | 0.8778 | 0.8620 | 0.8500 | 0.8743 | 3.5m |
| 6 | r3 | RoBERTa-base | 10 | 48 | 2e-5 | 128 | 0.2 | 0.8728 | 0.8555 | 0.8483 | 0.8629 | 2.6m |
| 7 | r1 | RoBERTa-base | 10 | 64 | 2e-5 | 64 | 0.3 | 0.8703 | 0.8514 | 0.8514 | 0.8514 | 1.9m |
| 8 | L3 | RoBERTa-large | 8 | 16 | 2e-5 | 128 | 0.2 | 0.8678 | 0.8490 | 0.8466 | 0.8514 | 8.2m |
| 9 | r2 | RoBERTa-base | 10 | 64 | 3e-5 | 64 | 0.3 | 0.8628 | 0.8493 | 0.8158 | 0.8857 | 1.6m |
| 10 | b2 | BERT-base | 15 | 64 | 2e-5 | 128 | 0.2 | 0.8404 | 0.8161 | 0.8208 | 0.8114 | 2.8m |
| 11 | b1 | BERT-base | 15 | 64 | 3e-5 | 128 | 0.2 | 0.8379 | 0.8094 | 0.8313 | 0.7886 | 2.2m |

### 关键发现

1. **模型规模是最强杠杆**: RoBERTa-large (+6.7% vs BERT) >> RoBERTa-base (+4.9%) > BERT-base
2. **大模型用小学习率**: RoBERTa-large 最佳 lr=1e-5，BERT 最佳 lr=3e-5
3. **max_len=128 稳定优于 64**: 推文虽短但 hashtag/@mention 可能截断关键信息
4. **Event 1 是瓶颈**: Ferguson 事件 recall 仅 50%，争议话题真假难辨，即使 Event Embedding 架构也未能根本解决

---

## 最终评估

### 整体指标 (val.csv, 401条)

| 指标 | 值 |
|------|:---:|
| Accuracy | **89.53%** (359/401) |
| Precision | 90.30% |
| Recall | 85.14% |
| F1 | 87.65% |
| FP（误报） | 16 |
| FN（漏报） | 26 |

### 各事件

| Event | 样本 | Acc | Prec | Rec | F1 | 备注 |
|-------|:---:|:---:|:---:|:---:|:---:|------|
| 0 | 13 | 92.3% | 1.00 | 0.83 | 0.91 | Gurlitt 艺术藏品 |
| 1 | 109 | 87.2% | 0.86 | 0.50 | 0.63 | ⚠️ Ferguson, recall 最低 |
| 2 | 1 | 100% | 1.00 | 1.00 | 1.00 | 极小样本 |
| 3 | 22 | 100% | 1.00 | 1.00 | 1.00 | 谣言主导 |
| 4 | 46 | 87.0% | 0.87 | 0.87 | 0.87 | 均衡分布 |
| 5 | 121 | 87.6% | 0.85 | 0.87 | 0.86 | 最大事件 |
| 6 | 89 | 93.3% | 0.94 | 0.94 | 0.94 | 最佳大事件 |

### 置信度分级

| 分级 | 样本数 | 准确率 |
|------|:---:|:---:|
| 确信（≥0.9） | 245 | 93.9% |
| 倾向（0.7–0.9） | 112 | 75.9% |
| 存疑（<0.7） | 44 | 54.5% |

确信级准确率 93.9%，说明模型高度自信时非常可靠；存疑级 54.5% 接近随机，表明模型知道何时不确定。

### 评估图表

运行 `evaluate.py` 后在 `figures/` 目录生成：

1. `confusion_matrix.png` — 混淆矩阵
2. `event_accuracy.png` — 各事件 Acc+F1（含整体基线）
3. `confidence_hist.png` — 置信度分布（正确/错误）
4. `calibration_curve.png` — 置信度校准曲线
5. `confidence_tiers.png` — 三级置信度准确率与占比
6. `event_f1_comparison.png` — 各事件 P/R/F1 对比

---

## 对抗训练

为提高模型对对抗样本的鲁棒性，训练了三个 RoBERTa-large 变体（均以 lr=1e-5、max_len=128、bs=16、dropout=0.2、seed=42 为基座）。

### 训练设置

| 模型 | 训练方式 | 训练时间 | 最佳 epoch | Val Acc |
|------|------|:---:|:---:|:---:|
| model_clean.pt | 正常微调，10 epoch | — | 9 | 89.53% |
| model_adv_v1.pt | 每 5 步注入 WordNet 同义词攻击（max_swaps=2），对抗权重 0.5 | ~11 min | 5 | 88.28% |
| model_adv_v2.pt | 每 2 步随机混合三种攻击（同义词/删词/字符交换），对抗权重 1.0 | ~28 min | 7 | 89.53% |

### V2 对抗训练曲线

```
Epoch  Train Loss  Train Acc  Val Loss  Val Acc  Val F1
1      0.9746      0.5940     0.4518    0.7756   0.7514
2      0.5740      0.8373     0.3592    0.8329   0.7886
3      0.3819      0.9046     0.3050    0.8653   0.8439
4      0.2558      0.9423     0.3429    0.8454   0.8000
5      0.1755      0.9644     0.3694    0.8853   0.8580
6      0.1088      0.9820     0.3727    0.8853   0.8631
7      0.0790      0.9845     0.3535    0.8953   0.8793  ← BEST
8      0.0794      0.9919     0.3905    0.8828   0.8669
9      0.0750      0.9937     0.4372    0.8728   0.8478
10     0.0363      0.9954     0.4291    0.8878   0.8696
11     0.0430      0.9979     0.4205    0.8928   0.8761
12     0.0404      0.9972     0.4310    0.8828   0.8614
```

### 多种子鲁棒性验证 (401条, CPU)

使用 WordNet 同义词替换（max_swaps=2），三个种子 (42/123/2026) 平均：

| 模型 | 干净 Acc | 平均翻转率 | 平均高置信翻转 |
|------|:---:|:---:|:---:|
| clean（干净训练） | 89.28% | 6.4% | 14.3 |
| adv_v1（同义词对抗） | 88.03% | 5.6% | 7.0 |
| adv_v2（多攻击对抗） | 88.28% | 5.7% | 9.3 |

### 结论

| 维度 | 最佳模型 | 说明 |
|------|:---:|------|
| 干净准确率 | **V2** | 89.53%，多攻击训练未伤泛化 |
| 攻击翻转率 | **V1/V2** | 平均 5.6–5.7%，均优于基线的 6.4% |
| 高置信翻转 | **V1** | 从 14.3 降至 7.0，最危险攻击减半 |
| 综合推荐 | **V2** | 准确率不掉 + 翻转率有效降低 + 多攻击均衡 |

---

## 关键设计决策

### Prompt 设计迭代

| 版本 | 方案 | LLM 角色 | 结果 |
|------|------|---------|------|
| v1 | 初始 | 被动解释者 | 可用，但解释较生硬 |
| v2 | 优化 prompt | 被动解释者 | 100% 成功率 |
| v3 | "独立法官" | 主动纠错 DL | ❌ 准确率降至 64.75%，过度纠正 |
| v4 | **回滚旧 prompt + 并行** | 被动解释者 | ✅ 89.53% + 解释质量好 |

### API 速率优化

SJTU API 官方限制 10 RPM。经 6 组参数压测（线程数 × 间隔），确定最优配置：**2 线程 + 6 秒间隔 = 10.3 RPM，零失败率**。401 条约 39 分钟，瓶颈在 API 限速非代码效率。`--no-llm` 模式约 4 分钟全程本地。

严格限速是因 V3 阶段发现 0.3s 间隔下 5 线程会触发限流导致重试阻塞，总耗时反而更长。

### 鲁棒性修复

- 延迟导入：`retrieval.py` / `llm_explainer.py` 仅在使用 LLM 时加载，`--no-llm` 不再需要安装 `sentence-transformers` 和 `openai`
- 自动创建输出目录：推理保存结果前检查并创建目录
- `--limit` 随机采样：避免前 N 条全是同一标签
- 模型路径守护：用户提供的路径缺失时 skip 而非崩溃
- Windows GBK 兼容：所有 Unicode 特殊字符替换为 ASCII 等效

### 数据泄漏检查

train.csv 2840 条 vs val.csv 401 条：id 完全不重复，仅 1 条文本巧合相同（不同 id），无泄漏。

---

## 已知问题

1. **LLM 推理慢**: DeepSeek API 官方限制 10 RPM，401 条需 ~39min。`--no-llm` 模式可快速验证分类器
2. **Event 1 弱项**: Ferguson 事件 recall 仅 50%，争议话题真假难辨
3. **模型文件大**: ~1.3GB/个，云盘共享，不提交 Git
4. **需 HF 网络**: 首次运行需下载 RoBERTa-large (~1.4GB)，国内建议 `HF_ENDPOINT=https://hf-mirror.com`

---

## 文件结构

```
├── README.md
├── report.pdf                 # 大作业报告
├── requirements.txt
├── .env.example               # API key 配置模板
├── .gitignore
│
├── rumer2026/                 # 原始数据
│   ├── train.csv              # 2840 条
│   └── val.csv                # 401 条
│
├── models/                    # 分类器模型
│   ├── __init__.py
│   ├── classifier.py          # RumorClassifier + load_model()（含 Event Embedding）
│   └── keyword_extractor.py   # predict() + 关键词提取
│
├── preprocess.py              # 数据预处理
├── retrieval.py               # 案例检索
├── llm_explainer.py           # LLM 解释生成（五要素 Prompt + 三级置信度提示）
├── event_context.py           # 7 个事件的背景文本
├── train.py                   # 训练脚本（支持正常/对抗训练模式）
│
├── inference.py               # 端到端推理管道（主入口）
├── evaluate.py                # 评估 + 6 张图表 + 置信度分级统计
├── adversarial.py             # 对抗样本攻防 + 三模型多种子对比
│
├── data/
│   └── index.pkl              # 检索索引（已提交，~5 MB）
├── checkpoints/               # 模型权重（Git 忽略，云盘分发）
├── results/                   # 推理输出（Git 忽略）
├── figures/                   # 评估图表（Git 忽略）
├── logs/                      # 训练日志（Git 忽略）
│
├── tools/                     # 辅助脚本
│   ├── benchmark_llm_rate.py  # LLM API 速率压测
│   ├── benchmark_seeds.py     # 多种子鲁棒性验证
│   ├── explore_events.py      # 数据探索
│   ├── analyze_results.py     # 推理结果分析
│   └── analyze_llm_verdict.py
│
├── server/                    # 服务器训练脚本（GPU 对抗训练）
│   ├── train_adv_v2.py        # 多攻击对抗训练
│   └── compare_robustness.py  # 全模型鲁棒性对比
│
└── tests/
    ├── test_retrieval.py
    └── test_llm_explainer.py
```

---

## 团队

| 成员 | 分工 |
|------|------|
| 韩宇飞 | 系统集成与端到端推理管道、评估脚本与图表生成、对抗攻防骨架搭建与服务器端训练、LLM 速率调优与鲁棒性修复、README 文档与最终报告撰写 |
| 姜新晨 | 数据预处理、BERT/RoBERTa 分类器训练与关键词提取、对抗样本攻防实验 |
| 靳卓达 | LLM 提示词工程、相似案例检索与解释生成、置信度分级与 Event Embedding 升级 |
