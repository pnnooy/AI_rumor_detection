# AI Rumor Detection — 可解释谣言检测系统

**2026《人工智能导论》课程大作业**

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

### 运行（老师一键复现）

```bash
# 1. 下载模型权重 + 检索索引（云盘链接见下方）
#    放置到: checkpoints/best_model.pt, data/index.pkl

# 2. 配置 API key（仅 LLM 模式需要）
cp .env.example .env
# 编辑 .env, 填入 SJTU_API_KEY

# 3. 推理
python inference.py --input rumer2026/val.csv --output results/val_results.csv --no-llm   # 仅分类 (~2min)
python inference.py --input rumer2026/val.csv --output results/val_results_full.csv         # 含 LLM 解释 (~5min)

# 4. 评估
python evaluate.py --input results/val_results.csv --output-dir figures/
```

### 下载链接

| 文件 | 大小 | 说明 |
|------|------|------|
| `checkpoints/best_model.pt` | 1.32 GB | RoBERTa-large 模型权重 |
| `data/index.pkl` | ~5 MB | 训练集检索索引 |

> 链接：[Rumor_detection — SJTU Pan](https://pan.sjtu.edu.cn/web/share/bb890ce0551de5fe239de6b8b1673e88)

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
| v4 | **3线程并行 + 0.6s 速率控制** | ~5min (未达预期, API 延迟瓶颈) |

> API 响应时间 ~2-3s/request 是瓶颈，客户端并行化效果有限。`--no-llm` 模式 ~2min 全程本地。

### Windows GBK 编码兼容

所有 `✓`/`⚠` 等 Unicode 字符替换为 `[OK]`/`[WARN]`，避免 Windows 终端 GBK 编码报错。

### SJTU API 内容审核

偶发 `inappropriate content` 拦截（Ferguson 推文含警察暴力词汇），频率约 1/401。程序自动捕获写入 `[LLM 调用失败: ...]`，不中断管道。

### 数据泄漏检查

train.csv 2840 条 vs val.csv 401 条: id 完全不重复, 仅 1 条文本巧合相同(不同id), 无泄漏。

---

## 已知问题

1. **LLM 推理慢**: DeepSeek API 响应 2-3s per request, 401 条需 ~5min。`--no-llm` 模式可快速验证分类器
2. **Event 1 弱项**: Ferguson 事件 recall 仅 50%, 争议话题真假难辨, 报告可讨论
3. **模型文件大**: 1.32GB, 云盘共享, 不提交 Git
4. **需 HF 网络**: 首次运行需下载 RoBERTa-large (~1.4GB), 国内建议 `HF_ENDPOINT=https://hf-mirror.com`

---

## 文件结构

```
├── README.md
├── .env.example              # API key 配置模板
├── .gitignore                # 排除 .env, checkpoints, data, results, figures
│
├── rumer2026/                # 原始数据
│   ├── train.csv             # 2840 条
│   └── val.csv               # 401 条
│
├── models/                   # 分类器模型 (AutoModel 通用版)
│   ├── __init__.py
│   ├── classifier.py         # RumorClassifier + load_model()
│   └── keyword_extractor.py  # predict() + _extract_keywords()
│
├── preprocess.py             # clean_text() + RumorDataset + create_dataloaders()
├── train.py                  # 训练脚本
│
├── retrieval.py              # CaseRetriever 类
├── llm_explainer.py          # LLMExplainer 类 + build_prompt()
├── event_context.py          # 7个事件的背景文本
│
├── inference.py              # 端到端推理脚本 (主入口)
├── evaluate.py               # 评估 + 5图表生成
├── adversarial.py            # 对抗样本模块
│
├── analyze_results.py        # 结果分析脚本
├── analyze_llm_verdict.py    # LLM 独立判断解析
│
├── checkpoints/              # 模型权重 (不提交Git)
├── data/                     # 检索索引 (不提交Git)
├── results/                  # 推理输出 (不提交Git)
├── figures/                  # 评估图表 (不提交Git)
├── logs/                     # 训练日志 (不提交Git)
│
├── server/                   # 服务器训练文件 (不上传GitHub)
│   ├── train.py, train_sweep_v2.py, run.sh, requirements.txt
│   ├── models/, rumer2026/
│   └── checkpoints/ (sweep_v2_summary.json)
│
└── tests/
    ├── test_retrieval.py
    └── test_llm_explainer.py
```

---

## 团队

| 成员 | 分工 | GitHub |
|------|------|--------|
| 姜新晨 | 数据预处理、BERT/RoBERTa 分类器训练、关键词提取 | — |
| 靳卓达 | LLM 提示词工程、相似案例检索、解释生成 | — |
| 韩宇飞 | 系统集成、评估、报告、对抗攻防、服务器训练 | — |

---

## 参考资料

- SJTU API: https://claw.sjtu.edu.cn/guide/sjtu-api/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Sentence Transformers: https://www.sbert.net/
