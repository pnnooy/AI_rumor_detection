# AI Rumor Detection — 可解释谣言检测系统

**2026《人工智能导论》课程大作业**

## 📑 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [训练历程](#训练历程)
- [最终评估](#最终评估)
- [调试与优化记录](#调试与优化记录)
- [已知问题](#已知问题)
- [文件结构](#文件结构)
- [Git 协作规范](#git-协作规范)
- [优化任务分工](#优化任务分工)
- [团队](#团队)
- [参考资料](#参考资料)

## 项目概述

构建**可解释谣言检测系统**：输入推文文本 + 事件ID，输出（1）谣言/非谣言分类，（2）判断依据的自然语言解释。

- **分类模型**: RoBERTa-large (355M), 微调, 验证集准确率 **89.53%**
- **解释模型**: DeepSeek-V3.2 via SJTU API, 中文 200 字解释, 五要素输入
- **检索模型**: sentence-transformers (all-MiniLM-L6-v2), 训练集语义相似度检索

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
pip install torch transformers sentence-transformers openai pandas numpy scikit-learn matplotlib seaborn tqdm python-dotenv nltk
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

> ⚠️ SJTU API 限速 10 次/分钟，全量 401 条约需 39 分钟，非代码效率问题。可选仅运行前 10 条进行快速测试。

```bash
cp .env.example .env        # 编辑 .env 填入 SJTU_API_KEY

# 快速体验：仅 10 条，约 1 分钟
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
              └──→ 中文解释 (200字)
```

### 技术栈

| 组件 | 技术 | 运行方式 |
|------|------|----------|
| 分类器 | RoBERTa-large (HuggingFace AutoModel) | 本地 CPU/GPU |
| 关键词提取 | Attention Weights (最后一层所有head平均) | 本地 |
| 案例检索 | sentence-transformers all-MiniLM-L6-v2 | 本地离线 |
| LLM 解释 | DeepSeek-V3.2 `deepseek-chat` via SJTU API | 远程 API |
| API 端点 | `https://models.sjtu.edu.cn/api/v1` | 仅校园网/VPN |

### LLM 解释为何不独立判断

经过实验验证（见调试记录），LLM 作为"独立法官"尝试纠正 DL 模型会导致准确率从 89% 降至 65%。原因：LLM 没有训练集标签知识，其常识判断与数据集标注标准不一致。最终方案为 LLM 作为 **DL 判断的透明解释者**，而非二审法官。

---

## 训练历程

### 模型演进

| 阶段 | 模型 | 最佳配置 | Val Acc | Val F1 | 提升 |
|------|------|------|:---:|:---:|:---:|
| 基线 | BERT-base (110M) | lr=2e-5, 5ep | 82.79% | 79.88% | — |
| V1 搜索 | BERT-base | lr=3e-5, max_len=128, 12ep | 85.04% | 82.56% | +2.25 |
| V2 搜索 | RoBERTa-base (125M) | lr=3e-5, max_len=128, 12ep | 87.78% | 86.20% | +4.99 |
| **⭐ V2 最佳** | **RoBERTa-large (355M)** | **lr=1e-5, max_len=128, 10ep** | **89.53%** | **87.65%** | **+6.74** |

### V2 完整实验结果（11组）

服务器: 4×3090 GPU (限定 GPU 4,5,6,7), HF_ENDPOINT=https://hf-mirror.com

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
2. **大模型用小学习率**: RoBERTa-large 最佳 lr=1e-5, BERT 最佳 lr=3e-5
3. **max_len=128 稳定优于 64**: 推文虽短但 hashtag/@mention 可能截断关键信息
4. **V1→V2 的 AutoModel 重构**: BERT 从原生 BertModel 切换到 AutoModel 后有 ~1% 轻微退化

---

## 最终评估

### 整体指标 (val.csv, 401条)

| 指标 | 值 |
|------|:---:|
| Accuracy | **89.53%** (359/401) |
| Precision | 90.30% |
| Recall | 85.14% |
| F1 | 87.65% |
| FP (误报) | 16 |
| FN (漏报) | 26 |

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

### 评估图表 (figures/)

1. `confusion_matrix.png` — 混淆矩阵
2. `event_accuracy.png` — 各事件 Acc+F1（含整体基线）
3. `confidence_hist.png` — 置信度分布（正确/错误）
4. `calibration_curve.png` — 置信度校准曲线
5. `event_f1_comparison.png` — 各事件 P/R/F1 对比

---

## 调试与优化记录

### Prompt 设计迭代

| 版本 | 方案 | LLM 角色 | 结果 |
|------|------|---------|------|
| v1 | 初始 | 被动解释者 | 可用, 但解释较生硬 |
| v2 | 旧 prompt (回滚) | 被动解释者 | 100% 成功率, 平均 206 字 |
| v3 | "独立法官" | 主动纠错 DL | ❌ 准确率降至 64.75%, 过度纠正 |
| v4 | **回滚旧 prompt + 并行** | 被动解释者 | ✅ 89.53% + 解释质量好 |

### API 速率优化

| 阶段 | 方案 | 401 条耗时 |
|------|------|:---:|
| 初始 | 串行, 每次新建 explainer | ~19min |
| v2 | 复用 explainer, 串行 | ~7-10min |
| v3 | 串行 + 0.75s 间隔 | ~19min (API 延迟主导) |
| v4 | **2线程并行 + 6s 速率控制** | ~39min (官方 10 RPM 稳定) |

### 最终速率配置

| 参数 | 值 | 说明 |
|------|:---:|------|
| 线程数 | 2 | SJTU API 官方限制 10 RPM |
| 调用间隔 | 6s | 保证不触发限流，零失败率 |
| 实测 RPM | 10.3 | 刚好贴满官限 |
| 401条耗时 | ~39 min | 瓶颈在 API 限速，非代码效率 |

> SJTU API 官方限制 10 RPM。经 bench 实测，2线程 6s 间隔 = 10.3 RPM 零失败，401条 ~39min。`--no-llm` 模式 ~4min 全程本地。

### Windows GBK 编码兼容

所有 `✓`/`⚠` 等 Unicode 字符替换为 `[OK]`/`[WARN]`，避免 Windows 终端 GBK 编码报错。

### SJTU API 内容审核

偶发 `inappropriate content` 拦截（Ferguson 推文含警察暴力词汇），频率约 1/401。程序自动捕获写入 `[LLM 调用失败: ...]`，不中断管道。

### 数据泄漏检查

train.csv 2840 条 vs val.csv 401 条: id 完全不重复, 仅 1 条文本巧合相同(不同id), 无泄漏。

---

## 对抗训练

为提高模型对对抗样本的鲁棒性，进行了两轮对抗训练。

### 训练环境

- 服务器: 4×3090 GPU (限定 GPU 4,5,6,7), HF_ENDPOINT=https://hf-mirror.com
- 数据: rumer2026/train.csv (2840条) + val.csv (401条)
- 基座: RoBERTa-large, lr=1e-5, max_len=128, batch_size=16, dropout=0.2, seed=42

### V1: 单攻击对抗训练 (checkpoints/adv_defense/)

| 参数 | 值 |
|------|:---:|
| 攻击类型 | WordNet 同义词替换 (max_swaps=2) |
| adv_weight | 0.5 |
| 注入频率 | 每 5 步 |
| 训练时间 | 11.0 分钟 |
| 最佳 epoch | 5 |

**结果:**

| 指标 | 原始模型 | 对抗V1 | 变化 |
|------|:---:|:---:|:---:|
| Val Accuracy | 89.53% | 88.28% | -1.25% |
| Val F1 | 87.65% | 85.89% | -1.76% |
| Precision | 90.30% | 90.51% | +0.21% |
| Recall | 85.14% | 81.71% | -3.43% |
| **攻击翻转率** | **6.0%** (24/401) | **4.1%** (4/98) | **-32%** |
| **高置信翻转** | **14 条** | **1 条** | **-93%** |

### V2: 多攻击混合对抗训练 (checkpoints/adv_v2/)

| 参数 | 值 |
|------|:---:|
| 攻击类型 | WordNet同义词(3) / 随机删词 / 字符交换 — 随机混合 |
| adv_weight | 1.0 |
| 注入频率 | 每 2 步 |
| 训练时间 | ~28 分钟 |
| 最佳 epoch | 7 |

训练曲线:
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

### 三模型综合对比 (200条 × 3种攻击, 服务器GPU实测)

| 模型 | 架构 | 干净Acc | 同义词 | 随机删词 | 字符交换 | 平均翻转 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| 原始 (best_model.pt) | BERT-base | 82.79% | 15.7% | 10.0% | 10.0% | 11.9% |
| 对抗V1 (adv_defense) | RoBERTa-large | 88.28% | **6.6%** | 6.0% | 4.0% | 5.5% |
| **对抗V2 (adv_v2)** | **RoBERTa-large** | **89.53%** | 8.1% | **4.0%** | **5.0%** | **5.7%** |

> ℹ️ 服务器原始模型为 BERT-base。本地 `best_model.pt` 为 RoBERTa-large (89.53%)。

### 本地最终验证 (401条, CPU, 多种子)

```bash
python adversarial.py --mode compare --original rumer2026/val.csv --seed 42
```

三个随机种子 (42/123/2026) 上的完整结果：

| 模型 | 干净Acc | flip(42) | flip(123) | flip(2026) | 平均翻转 | 平均高置信翻 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| clean (干净训练) | 89.28% | 6.4% | 6.4% | 6.4% | **6.4%** | 14.3 |
| adv_v1 (同义词对抗) | 88.03% | 6.4% | 5.3% | 5.1% | **5.6%** | 7.0 |
| adv_v2 (多攻击对抗) | 88.28% | 5.6% | 5.3% | 6.1% | **5.7%** | 9.3 |

### 结论

| 维度 | 最佳模型 | 说明 |
|------|:---:|------|
| 干净准确率 | **V2** | 88.28%，与原始仅差 1%，多攻击训练未伤泛化 |
| 攻击翻转率 | **V1/V2** | 平均 5.6-5.7%，均优于基线的 6.4% |
| 高置信翻转 | **V1** | 从 14.3 降至 7.0，最危险攻击减半 |
| 综合推荐 | **V2** | 准确率掉最少 + 翻转率有效降低 + 多攻击均衡 |

> **报告引用**: 三个随机种子(42/123/2026)上验证，翻转率从 6.4% 降至 5.6-5.7%，高置信翻转减半，结论稳健。

## 已知问题

1. **LLM 推理慢**: DeepSeek API 官方限制 10 RPM, 401 条需 ~39min。`--no-llm` 模式可快速验证分类器
2. **Event 1 弱项**: Ferguson 事件 recall 仅 50%, 争议话题真假难辨, 报告可讨论
3. **模型文件大**: 1.32GB, 云盘共享, 不提交 Git
4. **需 HF 网络**: 首次运行需下载 RoBERTa-large (~1.4GB), 国内建议 `HF_ENDPOINT=https://hf-mirror.com`

---

## 文件结构

```
├── README.md
├── .env.example              # API key 配置模板
├── .gitignore
│
├── rumer2026/                # 原始数据
│   ├── train.csv             # 2840 条
│   └── val.csv               # 401 条
│
├── models/                   # 分类器模型
│   ├── __init__.py
│   ├── classifier.py         # RumorClassifier + load_model()
│   └── keyword_extractor.py  # predict() + 关键词提取
│
├── preprocess.py             # 数据预处理
├── retrieval.py              # 案例检索
├── llm_explainer.py          # LLM 解释生成
├── event_context.py          # 7 个事件的背景文本
├── train.py                  # 训练脚本
│
├── inference.py              # 端到端推理管道 (主入口)
├── evaluate.py               # 评估 + 6 张图表
├── adversarial.py            # 对抗样本攻防 + 多模型对比
│
├── data/
│   └── index.pkl             # 检索索引 (已提交, ~5 MB)
├── checkpoints/              # 模型权重 (Git 忽略, 云盘分发)
├── results/                  # 推理输出 (Git 忽略)
├── figures/                  # 评估图表 (Git 忽略)
├── logs/                     # 训练日志 (Git 忽略)
│
├── tools/                    # 辅助工具脚本
│   ├── benchmark_llm_rate.py # LLM API 速率压测
│   ├── benchmark_seeds.py    # 多种子鲁棒性验证
│   ├── explore_events.py     # 数据探索
│   ├── analyze_results.py    # 推理结果分析
│   └── analyze_llm_verdict.py
│
├── server/                   # 服务器训练脚本
│   ├── train_adv_v2.py       # 多攻击对抗训练
│   └── compare_robustness.py # 全模型鲁棒性对比
│
└── tests/
    ├── test_retrieval.py
    └── test_llm_explainer.py
```

---

## Git 协作规范

### 分支结构

当前旧分支已废弃（落后 main 太多，且无独有内容）。从最新 main 新建分支：

```bash
# 姜新晨 — 对抗样本攻防
git checkout main && git pull origin main
git checkout -b jiang-xinchen/adversarial

# 靳卓达 — 置信度分级 + Event 升级
git checkout main && git pull origin main
git checkout -b jin-zhuoda/confidence-event
```

```
main  ← 主分支（集成 + 交付，只有韩宇飞可以合并）
  ├── jiang-xinchen/adversarial  ← 姜新晨
  └── jin-zhuoda/confidence-event ← 靳卓达
```

> **谁合并到 main**：韩宇飞负责 review 并 merge PR。姜新晨、靳卓达只 push 到自己的分支。

### 日常工作流

```bash
# 0. 拉取最新 main（每次开始工作前）
git checkout main && git pull origin main

# 1. 切到自己的分支
git checkout jiang-xinchen/adversarial

# 2. 改代码、测试、随时 commit
git add <文件>
git commit -m "feat: 新增 LLM 交叉验证防护"

# 3. push 到自己的远程分支
git push origin jiang-xinchen/adversarial          # 首次
git push origin jiang-xinchen/adversarial          # 后续

# 4. 完成一个阶段后，去 GitHub 发起 PR
```

### 提交格式

```bash
git commit -m "<type>: <简短描述>"

# type: feat(新功能) / fix(Bug) / refactor(重构) / docs(文档)
```

### 发起 PR

```
GitHub → Pull requests → New pull request
  base: main ← compare: jiang-xinchen/adversarial
  标题写清楚做了什么
  通知韩宇飞 Review → 由韩宇飞合并到 main
```

### ⚠️ 注意

| 做 | 别做 |
|----|------|
| 只 push 自己的分支 | 不要 push main |
| commit 前检查改动文件 | 不要 `git add .` 无脑全加 |
| 有问题问韩宇飞 | 不要提交 `.env` / `*.pt` / `*.pkl` |

---

## 🚀 优化任务分工 (2026-06-20)

> 主线已完成：RoBERTa-large 分类器 89.53%，端到端管道可运行。以下两个并行优化任务，完成后合并入 main。

---

### 任务一：对抗样本攻击与防护 🔴🛡️ — 姜新晨

> **评分关联**：老师明确表示"考虑对抗攻击的防护能力可以有额外加分"。对应评分中"创新性/鲁棒性"加分项。
> **目标**：证明系统在恶意扰动下不会轻易被欺骗，同时展现攻击+防护双重设计思想。

---

#### 📋 Phase 1：理解现有代码（15 分钟）

**必读文件（按顺序）**：

| 文件 | 关注点 |
|------|--------|
| `adversarial.py` | 攻击生成 `generate_adversarial()` + 鲁棒性分析框架 |
| `train.py` L131-170 | `--adversarial` 模式如何在训练中注入对抗样本 |
| `inference.py` L71-280 | 完整推理管道，理解输入输出格式 |
| `evaluate.py` | 评估脚本，理解 `results_csv` 的列结构 |

**关键理解**：
1. `adversarial.py` 的 `generate_adversarial()` 已能用 WordNet 同义词替换 1-2 个词，但 `analyze_robustness()` 是不完整的（到第 2 步就停了，提示用户手动运行 inference.py）
2. `train.py` 的对抗训练模式有一个 **bug**：L303 使用 `model.bert.resize_token_embeddings()`，但 `RumorClassifier` 的属性名是 `model.encoder`（因为升级到了 AutoModel），会导致 `AttributeError`
3. 当前缺少 LLM 交叉验证防护——这是你要新增的核心功能

---

#### 📋 Phase 2：修复 train.py 对抗训练 Bug（10 分钟）

**文件**：`train.py`

**Bug 位置**：L303
```python
# 当前（错误）：
model.bert.resize_token_embeddings(len(tokenizer))

# 修复为：
model.encoder.resize_token_embeddings(len(tokenizer))
```

**验证**：修复后运行 `python train.py --adversarial --epochs 1 --batch_size 16`，确认不报 `AttributeError`。

**同时检查** `RumorClassifier.__init__()` 是否也需要修改——当前 `models/classifier.py` 使用 `AutoModel`，属性名是 `self.encoder`，确认无误。

---

#### 📋 Phase 3：完善 adversarial.py — 端到端攻击评估（40 分钟）

**目标**：让 `analyze_robustness()` 能够**自动完成**整个攻防对比流程，不再需要用户手动操作。

**3.1 添加自动推理函数**

在 `adversarial.py` 中添加新函数 `run_inference_on_adversarial()`：

```python
def run_inference_on_adversarial(
    adversarial_csv: str,       # Phase 1 生成的对抗样本 CSV
    output_csv: str,            # 推理结果输出路径
    model_path: str = "checkpoints/best_model.pt",
    device: str = "cpu"
) -> pd.DataFrame:
    """
    用现有分类器对对抗样本做推理（仅 DL 分类，不调 LLM）
    
    步骤：
    1. 加载模型 load_model(model_path, device)
    2. 逐条调用 predict(text, event_id)
    3. 保存结果 CSV，包含列: id, text, event, true_label, pred_label, confidence, keywords
    4. 返回 DataFrame
    
    关键注意事项：
    - 必须按原始 event_id 传入 predict()，否则 [EVENT_N] token 会错配
    - 使用 --no-llm 等效逻辑（纯分类，不调 API）
    - 进度条用 tqdm 或每 50 条打印一次
    """
    # TODO: 实现
```

**实现提示**：参考 `inference.py` 的 Phase A 部分（L137-164），抽取其分类逻辑。核心循环：

```python
from models.classifier import load_model
from models.keyword_extractor import predict

model, tokenizer = load_model(model_path, device=device)

results = []
for i, (_, row) in enumerate(df.iterrows()):
    text = str(row['text'])
    event_id = int(row['event'])
    dl_result = predict(text, event_id)
    results.append({
        'id': row['id'],
        'text': text,
        'event': event_id,
        'true_label': int(row['label']),
        'pred_label': dl_result['label'],
        'confidence': dl_result['confidence'],
        'keywords': ','.join([w for w, _ in dl_result['keywords']]),
        'explanation': '',  # 对抗样本不需要 LLM 解释
    })
    if (i + 1) % 50 == 0:
        print(f"  推理进度: {i+1}/{len(df)}")
```

**3.2 增强 `analyze_robustness()` 为全自动流程**

重构 `analyze_robustness()`，将其从"打印提示"变为"自动执行"：

```python
def analyze_robustness(
    results_csv: str,          # 原始 val_results.csv
    original_csv: str,         # 原始 val.csv
    output_dir: str = "results",
    model_path: str = "checkpoints/best_model.pt",
    device: str = "cpu",
    auto_run_inference: bool = True  # 新增：自动运行推理
):
    """
    全自动鲁棒性分析管道：
    
    Step 1: 生成对抗样本 → {output_dir}/adversarial_samples.csv
    Step 2: 自动推理对抗样本 → {output_dir}/adversarial_results.csv
    Step 3: 对比原始 vs 对抗预测，计算翻转率
    Step 4: 按事件/置信度分组分析脆弱性
    Step 5: 保存分析报告 → {output_dir}/adversarial_report.txt
    
    所有步骤自动串联，无需人工干预。
    """
```

**3.3 增强 `compare_robustness()` 输出**

完善对比函数，额外输出：
- 翻转样本的原文 vs 对抗文本对照表（前 20 条）
- 哪些词被替换最频繁（WordNet 替换统计）
- 高置信度翻转（原 confidence > 0.9 但仍被翻转）——这是最危险的攻击

**3.4 新增：脆弱性分报告**

添加函数保存文本报告：

```python
def save_adversarial_report(..., output_dir: str):
    """保存 {output_dir}/adversarial_report.txt"""
    # 包含：
    # - 攻击成功率（整体 + 各事件）
    # - 高置信度翻转数量
    # - 最常被替换的词 Top 10
    # - 翻转方向分布 (0→1 vs 1→0)
```

---

#### 📋 Phase 4：实现 LLM 交叉验证防护（50 分钟）⭐ 新增核心功能

**目标**：当 DL 模型对原始文本和对抗文本的预测**不一致**时，调用 LLM 作为独立裁判进行二次判断。这是"防护"侧的亮点。

**4.1 在 `adversarial.py` 中添加 `llm_cross_validation()`**

```python
def llm_cross_validation(
    text: str,                          # 原推文文本
    dl_pred_label: int,                 # DL 模型预测 (可能是对抗后的)
    dl_confidence: float,               # DL 置信度
    original_pred_label: int,           # 原始文本的 DL 预测
    event_context: str = "",            # 事件背景
    explainer: "LLMExplainer | None" = None
) -> dict:
    """
    LLM 交叉验证：当 DL 预测不一致时，让 LLM 独立判断
    
    调用流程：
    1. 构造专门的法律/事实核查 prompt（不同于解释 prompt）
    2. LLM 基于事件背景 + 常识判断推文真实性
    3. 返回 {'llm_label': int, 'llm_reasoning': str, 'verdict': str}
    
    verdict 取值：
    - "支持DL"  — LLM 同意 DL 判断
    - "推翻DL"  — LLM 认为 DL 判断错误
    - "不确定"  — LLM 无法确定，建议人工复核
    
    关键设计：
    - prompt 必须强调 LLM 是独立判断，不受 DL 结果影响
    - temperature=0.1（比解释更确定）
    - 如果 LLM 调用失败 → verdict="不确定"（保守策略，不强行纠错）
    """
```

**4.2 LLM 交叉验证专用 Prompt 模板**：

```python
CROSS_VAL_PROMPT = """你是一个独立的谣言事实核查员。请仅根据以下信息判断这条推文是否在传播谣言。

[事件背景]
{event_context}

[推文内容]
"{text}"

[已知信息]
- 一个深度学习模型将这条推文判定为{"谣言" if dl_label == 1 else "非谣言"}（置信度 {confidence}）
- 但你可能面对的是被恶意修改过的推文文本（通过同义词替换等方式绕过检测）
- 请基于你的常识和对该事件背景的理解，独立判断这条推文是否在传播虚假信息

请用以下格式回复：
判断: [谣言/非谣言/不确定]
理由: [一句话说明判断依据]
"""
```

**4.3 在 `analyze_robustness()` 中集成 LLM 交叉验证**

在 Step 3（对比原始 vs 对抗预测）之后新增 Step 3.5：

```python
# Step 3.5: LLM 交叉验证（仅对翻转样本）
if use_llm_cross_validation and explainer is not None:
    flipped_mask = (original_results['pred_label'] != adversarial_results['pred_label'])
    flipped_indices = flipped_mask[flipped_mask].index
    
    for idx in flipped_indices:
        verdict = llm_cross_validation(
            text=original_texts[idx],
            dl_pred_label=adv_preds[idx],
            dl_confidence=adv_confs[idx],
            original_pred_label=orig_preds[idx],
            event_context=EVENT_CONTEXT.get(events[idx], ""),
            explainer=explainer
        )
        # 记录 LLM 是否成功防御了这次攻击
```

输出统计：
```
LLM 交叉验证结果 (针对 {n_flipped} 条翻转样本):
  支持DL判断:     X 条 (LLM 同意 DL 对抗后的判断 → 攻击可能有效)
  推翻DL判断:     Y 条 (LLM 纠正了对抗扰动 → 防御成功)
  不确定:          Z 条 (建议人工复核)
  防御成功率:      Y/(X+Y+Z)%
```

---

#### 📋 Phase 5：四组对照实验（30 分钟）

**实验设计**（在 `adversarial.py` 中新增 `run_defense_experiments()`）：

| 实验组 | 模型 | 对抗样本 | 防护机制 | 预期翻转率 |
|--------|------|:---:|------|:---:|
| **A 无防护** | best_model.pt | ✅ | 无 | 15-25% |
| **B 对抗训练** | 待训练 (--adversarial) | ✅ | 训练时注入对抗样本 | 8-15% |
| **C LLM交叉验证** | best_model.pt | ✅ | LLM 对翻转样本二次判断 | 5-10% |
| **D 双重防护** | 对抗训练模型 | ✅ | 对抗训练 + LLM 交叉验证 | 3-8% |

**实验 A**：直接对对抗样本推理（Phase 3 已完成）
**实验 B**：需要用 `--adversarial` 训练新模型
```bash
python train.py --adversarial --epochs 8 --lr 1e-5 --max_len 128 --batch_size 16 --save_dir checkpoints/adv_defense
```
**实验 C**：实验 A 的结果 + Phase 4 的 LLM 交叉验证
**实验 D**：实验 B 的模型 + LLM 交叉验证

**注意**：实验 B 需要 GPU（或 CPU 长时间运行）。如果没有 GPU，使用已有 `best_model.pt` 做实验 A/C 即可，实验 B/D 标记为"待验证"。

**输出对比表**：

```python
def print_defense_comparison(results: dict):
    """打印四组实验对比表"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║           对抗攻击与防护 — 四组实验对比              ║
    ╠══════════╦══════════╦══════════╦══════════╦══════════╣
    ║ 实验组   ║ 准确率   ║ 翻转率   ║ 高置信翻转║ 防护效果 ║
    ╠══════════╬══════════╬══════════╬══════════╬══════════╣
    ║ 原始     ║ 89.53%   ║ —       ║ —       ║ —       ║
    ║ A 无防护 ║ XX%      ║ XX%      ║ XX      ║ —       ║
    ║ B 对抗训练║ XX%     ║ XX%      ║ XX      ║ XX% ↓   ║
    ║ C LLM交叉║ XX%      ║ XX%      ║ XX      ║ XX% ↓   ║
    ║ D 双重   ║ XX%      ║ XX%      ║ XX      ║ XX% ↓↓  ║
    ╚══════════╩══════════╩══════════╩══════════╩══════════╝
    """)
```

---

#### 📋 Phase 6：CLI 入口统一（15 分钟）

在 `adversarial.py` 的 `main()` 中添加完整命令行接口：

```bash
# 完整攻防评估（一条命令跑完）
python adversarial.py \
    --input results/val_results.csv \
    --original rumer2026/val.csv \
    --mode full \
    --use-llm-defense \
    --output-dir results/adversarial

# 仅生成对抗样本
python adversarial.py --mode generate --original rumer2026/val.csv

# 仅评估（需要已有对抗样本推理结果）
python adversarial.py --mode evaluate \
    --input results/val_results.csv \
    --adversarial results/adversarial_results.csv
```

---

#### 📁 改动的文件清单

| 文件 | 改动类型 | 说明 |
|------|:---:|------|
| `adversarial.py` | **重写** | 新增自动推理、LLM交叉验证、四组实验对比 |
| `train.py` | **Bug修复** | L303: `model.bert` → `model.encoder` |

---

#### ✅ 验收标准

- [ ] `train.py --adversarial` 模式运行不报错
- [ ] `python adversarial.py --mode full --original rumer2026/val.csv` 一键完成攻击→推理→对比
- [ ] 输出四组实验对比数据（至少完成 A/C 两组；B/D 若有 GPU 也完成）
- [ ] LLM 交叉验证对翻转样本给出独立判断
- [ ] 生成 `results/adversarial/adversarial_report.txt` 文本报告
- [ ] 报告产出：对抗攻击成功率、防护机制效果、脆弱模式分析

---

#### 📝 报告产出要求

为 `report.pdf` 的"对抗攻击与防护分析"章节（3-4页）提供以下数据和文字：

1. **攻击方法描述**：WordNet 同义词替换，max_swaps=2，为什么选择这个方法（简单、白盒、贴合推文短文本）
2. **攻击效果**：翻转率 X%，其中高置信度翻转 N 条（最危险），典型翻转案例 3 个
3. **防护机制**：对抗训练原理 + LLM 交叉验证原理（附 prompt 设计思路）
4. **四组对比表**：如上表格
5. **结论**：哪个防护最有效？对抗训练 vs LLM 验证的权衡（成本 vs 效果）
6. **局限性讨论**：WordNet 同义词对推文语法（hashtag, @mention, 俚语）覆盖不全，真实攻击可能更多样

---

#### ⏱ 预估耗时：2-3 小时

| Phase | 内容 | 时间 |
|-------|------|:---:|
| Phase 1 | 阅读理解 | 15min |
| Phase 2 | Bug 修复 | 10min |
| Phase 3 | 完善 adversarial.py | 40min |
| Phase 4 | LLM 交叉验证 | 50min |
| Phase 5 | 四组实验 | 30min |
| Phase 6 | CLI + 测试 | 15min |

---

#### 🐛 常见问题预案

| 问题 | 原因 | 解决 |
|------|------|------|
| `generate_adversarial` 返回原文本 | 推文太短/全是俚语，没有可替换词 | 正常现象，标记为"不可扰动"，不计入翻转率分母 |
| WordNet 下载失败 | NLTK 数据未下载 | `python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"` |
| 对抗训练显存不足 | batch 太大 | 减小 `--batch_size` 到 8 或 4 |
| LLM 交叉验证返回"不确定"过多 | prompt 温度太低或事件背景不充分 | 调高 temperature 到 0.3，补充事件背景细节 |
| SJTU API 内容审核拦截 | Ferguson 推文含敏感词 | try/except 捕获，标记为 `[审核拦截]`，不计入统计数据 |
| `model.bert` AttributeError | AutoModel 重构后属性名变了 | 确认使用 `model.encoder` |

---

### 任务二：置信度分级解释 + Event 特征升级 🟡📊 — 靳卓达

> **评分关联**：置信度分级直接提升"可解释性"（15分），Event 特征升级提升"准确率"（15分）。一动两得。
> **目标**：让系统输出"知道何时不确定"的分级解释，同时用更强的 Event 信号修复 Event 1 recall=50% 的弱项。

---

#### 📋 背景知识

**当前状态**：
- Event 信息已通过 `[EVENT_N]` 特殊 token 注入分类器（`preprocess.py` L60）。这是**最简形式**——把 event 当一个普通 token 塞进文本。
- 置信度已输出但**没有分级展示**——用户看到 `confidence: 0.73` 但不知道这意味着什么。
- `llm_explainer.py` 内部有 `_get_confidence_level()` 方法用于 LLM prompt，但**用户看不到**这个分级。

**你要做的**：
1. 将 Event 从"一个 token"升级为"一个独立的 embedding 向量"，与文本表示拼接
2. 在推理输出、LLM 解释、评估图表中**显式展示**三级置信度

---

#### 📋 Phase 1：理解现有代码（15 分钟）

**必读文件（按顺序）**：

| 文件 | 关注点 |
|------|--------|
| `models/classifier.py` | `RumorClassifier.forward()` — 当前只用 [CLS] embedding 做分类 |
| `preprocess.py` L55-70 | `RumorDataset.__getitem__()` — 当前 `[EVENT_N] text` 拼接方式 |
| `llm_explainer.py` L57-72 | `_get_confidence_level()` — 已有分级逻辑但仅内部使用 |
| `inference.py` L137-164 | Phase A: 分类结果如何使用 event_id |
| `evaluate.py` L159-178 | `plot_confidence_histogram()` — 当前仅按正确/错误分组 |
| `event_context.py` | 7 个事件的背景文本 |

---

#### 📋 Phase 2：Event Embedding 升级（60 分钟）⭐ 核心改动

**问题**：当前 `[EVENT_N]` 只是一个特殊 token，嵌入在文本序列中。RoBERTa 的 self-attention 会让它和所有文本 token 交互——但 event 是**全局上下文**，不应该和单个词做 attention。更好的设计是：event 作为一个**独立的外部特征向量**，直接 concat 到 [CLS] 表示上。

**架构变更**：

```
当前架构：
  [EVENT_N] text → Encoder → [CLS] → classifier head → logits
  
新架构：
  text → Encoder → [CLS] embedding (hidden_dim)
  event_id → EventEmbedding Table (7 × event_emb_dim) → event_vec
  concat([CLS], event_vec) → classifier head → logits
```

**2.1 修改 `RumorClassifier.__init__()`**

在 `models/classifier.py` 中：

```python
class RumorClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        num_events: int = 7,           # 新增：事件数量
        event_emb_dim: int = 32,       # 新增：事件嵌入维度
        use_event_embedding: bool = True  # 新增：是否启用事件嵌入
    ):
        super().__init__()
        self.model_name = model_name
        self.use_event_embedding = use_event_embedding
        
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation='eager')
        self.hidden_size = self.encoder.config.hidden_size
        
        # 新增：事件嵌入表
        if use_event_embedding:
            self.event_embedding = nn.Embedding(num_events, event_emb_dim)
            self.event_emb_dim = event_emb_dim
            # 分类器输入维度 = [CLS] hidden + event embedding
            classifier_input_dim = self.hidden_size + event_emb_dim
        else:
            self.event_embedding = None
            classifier_input_dim = self.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
```

**2.2 修改 `RumorClassifier.forward()`**

```python
def forward(self, input_ids, attention_mask, event_ids=None, output_attentions=False):
    """
    Args:
        input_ids:      文本 token IDs (不含 event token)
        attention_mask: 文本 attention mask
        event_ids:      (batch,) 事件 ID tensor，每个样本一个整数 0-6
        output_attentions: 是否返回 attention weights（关键词提取用）
    """
    outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=output_attentions
    )
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]/<s>
    cls_embedding = self.dropout(cls_embedding)
    
    # 拼接事件嵌入
    if self.use_event_embedding and self.event_embedding is not None and event_ids is not None:
        event_vec = self.event_embedding(event_ids)  # (batch, event_emb_dim)
        combined = torch.cat([cls_embedding, event_vec], dim=-1)  # (batch, hidden+event_emb)
    else:
        combined = cls_embedding
    
    logits = self.classifier(combined)
    
    if output_attentions:
        return logits, outputs.attentions
    return logits
```

**2.3 修改 `RumorDataset.__getitem__()`** — 不再拼接 `[EVENT_N]` token

在 `preprocess.py` 中：

```python
def __getitem__(self, idx: int) -> dict:
    row = self.df.iloc[idx]
    text = clean_text(str(row['text']))
    event = int(row['event'])
    label = int(row['label'])
    
    # 变更：不再拼接 [EVENT_N]，直接编码纯文本
    # 旧: text_with_event = f"[EVENT_{event}] {text}"
    # 新: 纯文本编码 + event 单独传
    
    encoding = self.tokenizer(
        text, truncation=True, padding='max_length',
        max_length=self.max_len, return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.long),
        'event': event,  # 保留 event ID 传给 forward()
    }
```

**向后兼容考虑**：`create_dataloaders()` 中不再需要 `tokenizer.add_tokens(event_tokens)`。但 `load_model()` 中仍保留（旧 checkpoint 可能依赖）。新的 checkpoint 会在 `model_name` 字段旁保存 `use_event_embedding=True`。

**2.4 修改 `train.py` 的训练循环**

`train_epoch()` 和 `evaluate()` 中，调用 `model()` 时传入 `event_ids`：

```python
# 之前：
logits = model(input_ids, attention_mask, output_attentions=False)

# 之后：
event_ids = batch.get('event')  # 从 DataLoader 获取
if event_ids is not None and isinstance(event_ids, (list, torch.Tensor)):
    if not isinstance(event_ids, torch.Tensor):
        event_ids = torch.tensor(event_ids)
    event_ids = event_ids.to(device)
logits = model(input_ids, attention_mask, event_ids=event_ids, output_attentions=False)
```

**2.5 修改 `models/keyword_extractor.py`** — 适配新 forward 签名

`predict()` 函数中调用 `model()` 时也需要传入 `event_id`：

```python
# 之前：
logits, attentions = model(input_ids, attention_mask, output_attentions=True)

# 之后：
event_tensor = torch.tensor([event_id]).to(device)
logits, attentions = model(
    input_ids, attention_mask,
    event_ids=event_tensor, output_attentions=True
)
```

同时去掉 `predict()` 中的 `[EVENT_N]` 拼接逻辑。

**2.6 修改 `load_model()` — 兼容旧 checkpoint**

```python
def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_name = checkpoint.get('model_name', 'bert-base-uncased')
    
    # 读取新参数（旧 checkpoint 没有这些 key → 使用默认值 → 行为不变）
    use_event_embedding = checkpoint.get('use_event_embedding', False)
    num_events = checkpoint.get('num_events', 7)
    event_emb_dim = checkpoint.get('event_emb_dim', 32)
    
    model = RumorClassifier(
        model_name=model_name,
        num_classes=checkpoint.get('num_classes', 2),
        dropout=checkpoint.get('dropout', 0.3),
        num_events=num_events,
        event_emb_dim=event_emb_dim,
        use_event_embedding=use_event_embedding
    )
    # ... 其余不变
```

---

#### 📋 Phase 3：置信度分级解释系统（30 分钟）

**3.1 在 `inference.py` 中添加置信度分级标记**

在 `run_inference()` 的结果字典中加入 `confidence_level` 字段：

```python
def get_confidence_level(confidence: float) -> str:
    """返回置信度分级标签"""
    if confidence >= 0.9:
        return "确信"       # 系统高度自信，可直接采纳
    elif confidence >= 0.7:
        return "倾向"       # 系统倾向某判断，但建议注意
    else:
        return "存疑"       # 系统不确定，强烈建议人工复核
```

在 Phase A 的 `items` 构建中加入：
```python
items.append({
    # ... 原有字段 ...
    'confidence_level': get_confidence_level(dl_result['confidence']),
})
```

**3.2 在 `evaluate.py` 中添加置信度分级统计**

新增函数 `analyze_confidence_tiers(df)`：

```python
def analyze_confidence_tiers(df: pd.DataFrame):
    """
    按置信度分级统计准确率
    
    输出示例：
    分级      样本数    正确数    准确率
    确信(≥0.9)  245      230      93.9%
    倾向(0.7-0.9) 112    85       75.9%
    存疑(<0.7)  44       24       54.5%
    
    关键洞察：
    - 确信级别的准确率应显著高于整体 → 说明置信度可信
    - 存疑级别若准确率接近随机 → 说明 model 的自知之明有效
    """
    df = df.copy()
    df['tier'] = df['confidence'].apply(lambda c: 
        '确信(≥0.9)' if c >= 0.9 else ('倾向(0.7-0.9)' if c >= 0.7 else '存疑(<0.7)')
    )
    
    for tier in ['确信(≥0.9)', '倾向(0.7-0.9)', '存疑(<0.7)']:
        subset = df[df['tier'] == tier]
        if len(subset) > 0:
            acc = (subset['true_label'] == subset['pred_label']).mean()
            print(f"  {tier:<12} {len(subset):>5} 条   准确率: {acc:.1%}")
```

**3.3 新增置信度分级图表**

在 `evaluate.py` 中添加 `plot_confidence_tier_accuracy()`：

```python
def plot_confidence_tier_accuracy(df: pd.DataFrame, output_dir: str):
    """
    三级置信度的准确率柱状图 + 占比饼图
    
    左子图：三个柱（确信/倾向/存疑），每柱上标注准确率和样本数
    右子图：饼图显示三级样本量占比
    """
    # 两个子图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ... 实现
    plt.savefig(f'{output_dir}/confidence_tiers.png', dpi=150, bbox_inches='tight')
```

**3.4 在 LLM prompt 中增强分级信息**

修改 `llm_explainer.py` 的 `build_prompt()`，在"模型判断结果"部分强调分级：

```python
# 当前 prompt 已有 conf_desc，但不够醒目。增强为：
[模型判断结果]
判定: {label_str}
置信度: {confidence:.0%} — {conf_desc}
{'⚠️ 注意：置信度较低，此判断可能有误，请重点复核。' if confidence < 0.7 else ''}
{'ℹ️ 提示：置信度中等，建议结合其他信息综合判断。' if 0.7 <= confidence < 0.9 else ''}
{'✅ 高置信判断，可以直接采纳。' if confidence >= 0.9 else ''}
```

---

#### 📋 Phase 4：在自己电脑上训练 + 评估对比（60 分钟）

**重要：所有训练在你自己的电脑上完成，不依赖服务器。**

**4.1 为什么可以在个人电脑上训练**

| 方案 | 模型 | 参数量 | CPU 预估 | 用途 |
|------|------|:---:|------|------|
| **快速验证** | `bert-base-uncased` | 110M | ~15min/epoch | 验证代码 pipeline 正确 |
| **正式训练** | `roberta-base` | 125M | ~25min/epoch | 拿到可对比的正式结果 |

> RoBERTa-large (355M) 在 CPU 上太慢（~2h/epoch），不适合个人电脑。用 roberta-base 做正式训练，效果对比结论同样有效——我们对比的是 **Event Embedding 架构 vs [EVENT_N] token 架构**，不是模型大小。

**4.2 第一步：快速验证（5 分钟，确保代码能跑通）**

```bash
# 用 bert-base-uncased 跑 1 个 epoch，验证 pipeline 不报错
python train.py \
    --bert_model bert-base-uncased \
    --epochs 1 \
    --batch_size 8 \
    --lr 2e-5 \
    --max_len 64 \
    --dropout 0.3 \
    --save_dir checkpoints/smoke_test \
    --device cpu \
    --seed 42
```

验证点：
- 数据加载正常（2840 条训练 + 401 条验证）
- 训练循环不报错（特别是 `event_ids` 传入 `forward()` 正常）
- 模型保存成功（`checkpoints/smoke_test/best_model.pt` 存在）

如果 1 epoch CPU 太慢，可以把训练集缩减到 200 条快速验证（临时改 `train.csv` 路径指向一个截断版本）。

**4.3 第二步：正式训练（用 roberta-base，约 2-3 小时）**

```bash
# 方案 A：CPU 训练（任何电脑都能跑，就是慢一点）
python train.py \
    --bert_model roberta-base \
    --epochs 5 \
    --batch_size 8 \
    --lr 2e-5 \
    --max_len 128 \
    --dropout 0.2 \
    --save_dir checkpoints/event_embedding \
    --device cpu \
    --seed 42

# 方案 B：如果有 NVIDIA 显卡（检查: python -c "import torch; print(torch.cuda.is_available())"）
python train.py \
    --bert_model roberta-base \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_len 128 \
    --dropout 0.2 \
    --save_dir checkpoints/event_embedding \
    --device cuda \
    --seed 42
```

**如果 CPU 太慢的替代方案**：
- 减少 epoch 到 3（够看对比趋势了）
- 减少 `max_len` 到 64
- 只用 CPU 跑 1-2 个 epoch 拿到初步结果，报告里标"初步实验"

**4.4 对比评估**

训练完成后，用**相同的 roberta-base 配置但关闭 event_embedding** 训练一个对照组（baseline for comparison）：

```bash
# 对照组：同样的 roberta-base，但不启用 event embedding
# 做法：临时的 RumorClassifier(use_event_embedding=False) + 保留 [EVENT_N] token 拼接
# 或者在 train.py 里加一个 --no-event-embedding 参数来切换
```

对比输出：

| 指标 | 旧架构 ([EVENT_N] token) | 新架构 (Event Embedding) | 变化 |
|------|:---:|:---:|:---:|
| Overall Acc | ? | ? | ? |
| Overall F1 | ? | ? | ? |
| **Event 1 Recall** | ? | ? | **核心指标** |
| Event 1 Acc | ? | ? | ? |
| 训练时间 | ? | ? | ? |

**如果 Event 1 recall 没有提升也不要紧**——这是有价值的"负面结果"，可以在报告中讨论"Event Embedding 并非银弹，跨事件泛化的根本挑战在于事件本身的争议性"，同样是好内容。

**4.5 置信度分级统计对比**

用训练好的新模型跑推理：

```bash
python inference.py \
    --input rumer2026/val.csv \
    --output results/val_event_embedding.csv \
    --model checkpoints/event_embedding/best_model.pt \
    --no-llm
```

然后对比新旧模型的置信度分级分布：

| 分级 | 旧模型占比 | 旧模型准确率 | 新模型占比 | 新模型准确率 |
|------|:---:|:---:|:---:|:---:|
| 确信(≥0.9) | ?% | ?% | ?% | ?% |
| 倾向(0.7-0.9) | ?% | ?% | ?% | ?% |
| 存疑(<0.7) | ?% | ?% | ?% | ?% |

---

#### 📋 Phase 5：推理管道更新（20 分钟）

**5.1 更新 `inference.py` 适配 event_ids**

Phase A 中调用 `predict()` 时，确保传入 event_id（当前已传，检查签名是否一致）。

**5.2 新增 `--output-confidence-tiers` 参数**

```bash
python inference.py --input val.csv --output results.csv --no-llm --output-confidence-tiers
```

当此标志开启时，输出 CSV 额外包含 `confidence_level` 列，并在控制台打印分级统计。

**5.3 更新 `evaluate.py` 集成新图表**

在 `evaluate()` 主函数中加入 Phase 3.3 的新图表调用：

```python
# 在 evaluate() 的绘图步骤中加入
if 'confidence' in df.columns:
    plot_confidence_tier_accuracy(df, output_dir)  # 新增
```

---

#### 📁 改动的文件清单

| 文件 | 改动类型 | 说明 |
|------|:---:|------|
| `models/classifier.py` | **重写** | 新增 EventEmbedding + 修改 forward 签名 |
| `preprocess.py` | **修改** | 去掉 [EVENT_N] 拼接，event 单独输出 |
| `models/keyword_extractor.py` | **修改** | 适配新 forward(event_ids=)，去掉 [EVENT_N] 拼接 |
| `train.py` | **修改** | 训练循环传入 event_ids，适配新 RumorClassifier 参数 |
| `inference.py` | **修改** | 加入置信度分级字段，新增 --output-confidence-tiers |
| `evaluate.py` | **新增函数** | analyze_confidence_tiers() + plot_confidence_tier_accuracy() |
| `llm_explainer.py` | **修改** | 增强 prompt 中的置信度分级显示 |

---

#### ✅ 验收标准

- [ ] `RumorClassifier(use_event_embedding=True)` 可以正常前向传播
- [ ] `python train.py` 用新架构训练不报错，val_acc ≥ 旧模型
- [ ] Event 1 recall 从 50% 提升到 ≥ 60%（核心指标）
- [ ] `inference.py` 输出包含 `confidence_level` 列
- [ ] `evaluate.py` 输出三级置信度准确率统计表
- [ ] 新图表 `confidence_tiers.png` 生成正常
- [ ] LLM 解释中低置信度样本有明确的复核提醒
- [ ] 旧 checkpoint（无 event_embedding）仍可加载（向后兼容）
- [ ] 报告产出：Event 特征升级前后对比数据 + 置信度分级分析

---

#### 📝 报告产出要求

为 `report.pdf` 提供以下内容：

1. **Event Embedding 设计**：架构图（ASCII即可），解释为什么独立 embedding 优于 text token（全局上下文、不干扰 attention、维度可控）
2. **消融对比表**：[EVENT_N] token vs Event Embedding 的各项指标
3. **Event 1 改善分析**：Ferguson 事件 recall 提升的原因分析（event embedding 让模型学会"Ferguson争议大→更倾向谣言"的先验）
4. **置信度分级**：三级统计表 + `confidence_tiers.png` + 讨论（确信级准确率是否显著高于整体？存疑级是否合理？）
5. **可解释性提升**：分级解释如何帮助用户理解系统判断——不是"89%准确"一句话，而是"在它确定的时候你可以信它，在它不确定的时候你应该复核"

---

#### ⏱ 预估耗时：2.5-3.5 小时

| Phase | 内容 | 时间 |
|-------|------|:---:|
| Phase 1 | 阅读理解 | 15min |
| Phase 2 | Event Embedding 升级 | 60min |
| Phase 3 | 置信度分级系统 | 30min |
| Phase 4 | 自己电脑训练+对比 | 60min |
| Phase 5 | 管道更新+测试 | 20min |
| 训练等待 | CPU 训练 roberta-base 约 2-3h（可后台跑） | — |

---

#### 🐛 常见问题预案

| 问题 | 原因 | 解决 |
|------|------|------|
| `forward()` 收到 None event_ids | 旧 DataLoader 没传 event | 检查 `RumorDataset.__getitem__` 返回的 'event' 字段 |
| 新模型 val_acc 低于旧模型 | event_emb_dim 不合适或训练不充分 | 尝试 event_emb_dim=16/64，增加 epochs |
| 旧 checkpoint 加载报错 | 旧 checkpoint 没有 `use_event_embedding` key | `checkpoint.get('use_event_embedding', False)` 默认 False |
| keyword_extractor 报错 | predict() 还在拼接 [EVENT_N] | 去掉拼接逻辑，改为传入 event_ids 参数 |
| 置信度分级"确信"占比过高 | 模型过拟合，几乎所有预测 confidence > 0.9 | 调高 dropout 或减少训练 epoch |
| Event 1 recall 没有提升 | event embedding 不是银弹，Ferguson 本身争议大 | 报告讨论"跨事件泛化的根本局限"也是好内容 |

---

## 团队

| 成员 | 分工 | GitHub |
|------|------|--------|
| 姜新晨 | 数据预处理、BERT/RoBERTa 分类器训练、关键词提取、对抗样本攻防 | — |
| 靳卓达 | LLM 提示词工程、相似案例检索、解释生成、置信度分级+Event特征升级 | — |
| 韩宇飞 | 系统集成、评估、报告、对抗攻防骨架、服务器训练、项目管理 | — |

---

## 参考资料

- SJTU API: https://claw.sjtu.edu.cn/guide/sjtu-api/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Sentence Transformers: https://www.sbert.net/
