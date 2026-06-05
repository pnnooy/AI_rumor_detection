# AI Rumor Detection — 可解释谣言检测系统

**2026《人工智能导论》课程大作业**

---

## 项目概述

构建一个**可解释的谣言检测系统**：输入一条推文（tweet），系统输出（1）是否为谣言，（2）判断依据的自然语言解释。

- **任务类型**: 文本二分类 + 自然语言解释生成
- **数据集**: 2840 条训练推文 + 401 条验证推文，7 个事件主题
- **核心要求**: 深度学习模型做分类 + 大语言模型做可解释性分析

---

## 数据说明

| 项目 | 值 |
|------|-----|
| 训练集 | 2840 条推文，7 个事件 |
| 验证集 | 401 条推文 |
| 文本平均长度 | 16.3 词（最短 3 词） |
| 标签 | 0 = 非谣言, 1 = 谣言 |
| 谣言占比 | 训练集 43.7%, 验证集 43.6% |

### 事件分布

| Event | 训练集 | 验证集 | 谣言占比 | 说明 |
|-------|--------|--------|----------|------|
| 0 | 66 | 13 | ~20% | 小样本 |
| 1 | 799 | 109 | ~25% | 大样本 |
| 2 | 9 | 1 | ~100% | **极端小样本** |
| 3 | 162 | 22 | ~99% | 几乎全是谣言 |
| 4 | 327 | 46 | ~51% | 均匀分布 |
| 5 | 854 | 121 | ~43% | 最大样本 |
| 6 | 623 | 89 | ~53% | 大样本 |

> ⚠️ Event 2（仅 9 条全谣言）和 Event 3（99% 谣言）是极端情况，需要特别关注。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        推文输入                                    │
│         "Witness says police fired tear gas at protesters"        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
   ┌────────────────┐        ┌──────────────────┐
   │  BERT 分类器    │        │  相似案例检索      │
   │  (PyTorch)     │        │  (sentence-       │
   │                │        │   transformers)   │
   │ 输出:          │        │                   │
   │ · label (0/1)  │        │ 输出:             │
   │ · confidence   │        │ · Top-3 相似推文   │
   │ · top-k 关键词  │        │ · 对应真实标签     │
   └───────┬────────┘        └────────┬─────────┘
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   LLM 解释生成器       │
          │   (DeepSeek-V3.2      │
          │    通过 SJTU API)      │
          │                       │
          │  输入:                 │
          │  · 原推文文本           │
          │  · DL 预测 + 置信度     │
          │  · 模型关注的关键词     │
          │  · 相似案例参考         │
          │  · 事件背景信息         │
          │                       │
          │  输出:                 │
          │  · 自然语言解释文本      │
          └───────────┬───────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                      最终输出                                     │
│  {                                                               │
│    "label": 1,                                                   │
│    "confidence": 0.87,                                           │
│    "explanation": "该推文被判定为谣言，主要依据：1. 使用匿名信源   │
│                    'witness says'，无法验证...",                   │
│    "keywords": ["witness", "says", "fired"]                      │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 组件 | 技术 | 运行方式 |
|------|------|----------|
| DL 分类器 | BERT-base-uncased (HuggingFace Transformers) | 本地训练/推理 |
| 关键词提取 | BERT Attention Weights | 本地推理 |
| 相似案例检索 | sentence-transformers (all-MiniLM-L6-v2) | 本地离线 |
| LLM 解释 | DeepSeek-V3.2 via SJTU API | 远程 API 调用 |
| 框架 | PyTorch 2.x + OpenAI SDK | — |

### LLM 模型选型

SJTU API 提供 5 个模型，最终选择 `deepseek-chat`（DeepSeek V3.2）：

| 模型 | 判断 | 理由 |
|------|------|------|
| **DeepSeek V3.2** `deepseek-chat` | ✅ 选用 | 685B 参数，通用文本最强，中文输出好，指令跟随稳定，32k 上下文对本任务完全够用 |
| DeepSeek Reasoner `deepseek-reasoner` | ❌ | 深度推理模式更适合数学/逻辑题，生成解释不需要额外的"思考链"，反而更慢更费 token |
| MiniMax-M2.7 `minimax` | ❌ | 230B 参数，偏智能体/工具调用，与我们的结构化解释生成任务不匹配 |
| GLM-5.1 `glm` | ❌ | 754B 参数最大但侧重代码与超长文本，杀鸡用牛刀，中文解释质量不如 DeepSeek 稳定 |
| Qwen3.5-27B `qwen` | ❌ | 仅 27B，参数最小，解释质量和一致性可能不足；强项是视觉多模态，本任务用不上 |

> **结论**：`deepseek-chat` 最适合"给定结构化信息 → 输出规整中文解释"这个任务，参数够大、中文好、速度快，且天然满足 API 对 user 角色的要求。

---

## 分工与详细工作指南

### 👤 成员 A：数据预处理 + BERT 分类器

**职责**: 把原始数据变成能用的形式，训练并评估 DL 分类模型。

#### 1. 数据预处理

**输入**: `rumer2026/train.csv`, `rumer2026/val.csv`

**处理步骤**:
- [ ] 文本清洗：去除 URL (`http://t.co/...`)、特殊字符、多余空格
- [ ] 保留 hashtag 文本（去掉 `#` 符号保留词，如 `#Ferguson` → `Ferguson`）
- [ ] 保留 `@` 提及（可选：保留用户名或统一替换为 `@USER`）
- [ ] Tokenization：使用 BERT tokenizer（`bert-base-uncased`）
- [ ] 将 event_id 编码为特殊 token（格式：`[EVENT_0]` ~ `[EVENT_6]`）
- [ ] 构建 PyTorch Dataset 类，返回 `input_ids`, `attention_mask`, `label`

**输出文件**:
```
data/
  processed_train.pt  或  DataLoader 直接使用
  processed_val.pt
```

**代码模板**:
```python
# preprocess.py
import pandas as pd
import re
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class RumorDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def clean_text(self, text):
        # 去除 URL
        text = re.sub(r'http\S+', '', text)
        # 保留 hashtag 词
        text = re.sub(r'#(\w+)', r'\1', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.clean_text(row['text'])
        event = int(row['event'])
        # 拼接 event token
        text = f"[EVENT_{event}] {text}"
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.df)
```

**验收标准**:
- `clean_text()` 处理后文本可读，无 URL 残留
- Dataset 可正常迭代，batch size=16 不报错
- event token 正确拼接到文本开头

---

#### 2. BERT 分类器训练

**模型结构**:
```python
import torch.nn as nn
from transformers import BertModel

class RumorClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_embedding)
        
        if output_attentions:
            return logits, outputs.attentions  # 用于关键词提取
        return logits
```

**训练配置**:
| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 5-10 | 看验证集 loss 早停 |
| Batch size | 16 或 32 | 取决于 GPU 内存 |
| Learning rate | 2e-5 | BERT 标准微调学习率 |
| Optimizer | AdamW | 带权重衰减 |
| Scheduler | Linear warmup | 前 10% steps 线性升温 |
| Max length | 64 | 数据最长 31 词，64 足够 |
| Loss | CrossEntropyLoss | 标准分类损失 |
| Seed | 42 | 所有随机数固定 |

**训练脚本要求**:
- `train.py` 可独立运行
- 每 epoch 结束后在验证集上评估
- 保存最佳模型到 `checkpoints/best_model.pt`
- 训练日志输出到 `logs/training.log`
- 使用 `tqdm` 显示进度条

**输出**:
- `checkpoints/best_model.pt` — 模型权重文件
- `logs/training.log` — 训练日志（loss, accuracy per epoch）

**验收标准**:
- 验证集准确率 ≥ 75%（基线）
- 模型文件可被 `torch.load()` 正常加载
- 训练结束后自动输出验证集评估报告（accuracy, precision, recall, F1）

---

#### 3. 关键词提取（Attention 权重）

**任务**: 从 BERT 最后一层的 attention weights 中提取模型最关注的词。

**方法**:
```python
def extract_keywords(model, tokenizer, text, event_id, top_k=5):
    """
    提取 BERT 模型在分类时最关注的 top_k 个词
    
    Returns:
        list of (word, attention_score) tuples
    """
    # 拼接 event token
    text_with_event = f"[EVENT_{event_id}] {text}"
    inputs = tokenizer(text_with_event, return_tensors='pt')
    
    # 前向传播，获取 attention
    logits, attentions = model(
        inputs['input_ids'],
        inputs['attention_mask'],
        output_attentions=True
    )
    
    # attentions: tuple of (batch, num_heads, seq_len, seq_len)
    # 取最后一层，所有头平均，取 [CLS] token 对其他 token 的注意力
    last_layer_attn = attentions[-1]  # (batch, num_heads, seq_len, seq_len)
    cls_attn = last_layer_attn[:, :, 0, :].mean(dim=1)  # (batch, seq_len)
    
    # 获取每个 token 的注意力权重（排除 [CLS], [SEP], [PAD]）
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_scores = []
    for i, (token, score) in enumerate(zip(tokens, cls_attn[0])):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            token_scores.append((token, score.item()))
    
    # 按分数排序，取 top_k
    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_k]
```

**输出格式**（供成员 B 使用）:
```python
# 函数返回格式
keywords = [
    ("witness", 0.23),
    ("says", 0.16),
    ("fired", 0.11),
    ("protesters", 0.07),
    ("police", 0.05)
]
```

**验收标准**:
- 对任意输入文本，能返回 top-5 关键词及对应的 attention score
- 关键词是可读的英文单词（非 subword token）
- 分数归一化到 0-1 之间

---

#### 成员 A 交付清单

```
models/
  classifier.py          # RumorClassifier 模型定义
  keyword_extractor.py   # 关键词提取函数
train.py                 # 训练脚本（可独立运行）
preprocess.py            # 数据预处理 + Dataset 类
checkpoints/
  best_model.pt          # 训练好的模型权重
logs/
  training.log           # 训练日志
```

**对外接口**（供成员 C 集成）:
```python
# 推理接口
def predict(text: str, event_id: int) -> dict:
    """
    Returns:
        {
            "label": 0 或 1,
            "confidence": float (0-1),
            "keywords": [("word", score), ...]
        }
    """
```

---

### 👤 成员 B：LLM 解释生成 + 相似案例检索

**职责**: 构建相似案例检索系统，设计 LLM 提示词，生成高质量的自然语言解释。

#### 1. 相似案例检索

**方法**: 使用 sentence-transformers 将训练集全部推文编码为向量，推理时计算余弦相似度，返回最相似的 3 条。

**实现**:
```python
# retrieval.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle

class CaseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None   # 训练集全部向量
        self.texts = None        # 训练集全部文本
        self.labels = None       # 训练集全部标签
        self.events = None       # 训练集全部事件
    
    def build_index(self, csv_path):
        """预处理：编码训练集全部推文"""
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.events = df['event'].tolist()
        self.embeddings = self.model.encode(
            self.texts, 
            show_progress_bar=True,
            batch_size=64
        )
        # 保存索引
        self._save_index('data/index.pkl')
    
    def search(self, query_text, top_k=3):
        """检索最相似的 top_k 条训练推文"""
        query_vec = self.model.encode([query_text])
        similarities = np.dot(self.embeddings, query_vec.T).squeeze()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'label': self.labels[idx],
                'event': self.events[idx],
                'similarity': float(similarities[idx])
            })
        return results
```

**输出格式**（供 LLM 提示词使用）:
```python
similar_cases = [
    {
        "text": "Eyewitness claims police used excessive force",
        "label": 1,           # 1 = 谣言
        "event": 5,
        "similarity": 0.89
    },
    {
        "text": "Anonymous source reports police misconduct",
        "label": 1,
        "event": 5,
        "similarity": 0.85
    },
    {
        "text": "Official statement confirms police followed protocol",
        "label": 0,           # 0 = 非谣言
        "event": 5,
        "similarity": 0.78
    }
]
```

**验收标准**:
- 首次运行 `build_index()` 后生成本地索引文件，后续加载不重复编码
- `search()` 返回结果中包含不同标签的案例（非全谣言或全非谣言）
- 检索延迟 < 0.5 秒/条

---

#### 2. LLM 解释生成

**模型**: DeepSeek-V3.2 (`deepseek-chat`)，通过 SJTU API 调用

**API 调用方式**（OpenAI 兼容格式）:
```python
# llm_explainer.py
from openai import OpenAI
import os

class LLMExplainer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SJTU_API_KEY"),
            base_url="https://models.sjtu.edu.cn/api/v1"
        )
        self.model = "deepseek-chat"
```

> ⚠️ **API 要求**：
> - 仅限校园网访问（校外需通过 SJTU VPN）
> - 请求中必须包含 `user` 角色的消息，否则不返回内容
> - 速率限制：100 次/分钟，100000 tokens/分钟，10亿 tokens/周
> - API key 有效期至 2026-06-30
    
    def build_prompt(self, text, dl_result, similar_cases, event_info):
        """构建包含五要素的提示词"""
        label_str = "谣言" if dl_result['label'] == 1 else "非谣言"
        confidence = dl_result['confidence']
        keywords = [w for w, s in dl_result['keywords']]
        
        # 置信度分级描述
        if confidence > 0.9:
            conf_desc = "高度确信"
        elif confidence > 0.7:
            conf_desc = "倾向于判定"
        else:
            conf_desc = "勉强判定，建议人工复核"
        
        # 相似案例格式化
        cases_str = ""
        for i, case in enumerate(similar_cases):
            case_label = "谣言" if case['label'] == 1 else "非谣言"
            cases_str += f"""
案例{i+1}: "{case['text']}"
  → 真实标签: {case_label} | 相似度: {case['similarity']:.2f}
"""
        
        prompt = f"""你是一个谣言检测系统的解释模块。系统已经对一条社交平台推文做出了自动判断，你需要帮助用户理解判断依据。

[推文内容]
"{text}"

[事件背景]
{event_info}

[模型判断结果]
判定: {label_str}
置信度: {confidence:.0%}
系统对此判断{conf_desc}

[模型关注的关键词（按重要性排序）]
{', '.join(keywords)}

[训练集中最相似的案例]
{cases_str}

请输出中文解释（200字以内），包含以下要点：
1. 这条推文为什么被判定为谣言/非谣言？结合关键词和文本线索分析
2. 判断的可信度如何？如果置信度较低，应坦诚说明不确定性
3. 有什么需要人工复核的地方吗？
"""
        return prompt
    
    def explain(self, text, dl_result, similar_cases, event_info):
        """生成解释"""
        prompt = self.build_prompt(text, dl_result, similar_cases, event_info)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的谣言分析助手。请基于给定的分析信息，用中文输出清晰、有理有据的解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,      # 低温度保证解释一致性
            max_tokens=512
        )
        
        return response.choices[0].message.content
```

**提示词设计原则**:
1. ✅ 告诉 LLM 它的角色（系统解释模块）
2. ✅ 提供结构化输入（五要素明确分段）
3. ✅ 约束输出格式和长度
4. ✅ 要求基于给定信息，不瞎编
5. ✅ 低 temperature 保证输出一致

**验收标准**:
- `.env` 文件配置 API key 后可正常调用
- 单次解释延迟 < 5 秒
- 解释文本 150-300 字，中文，可读性好
- 解释内容引用了给定的关键词和案例

---

#### 3. 事件背景数据（供 LLM 使用）

为每个事件准备一段简短背景描述：

```python
# event_context.py
EVENT_CONTEXT = {
    0: "关于 Gurlitt 艺术藏品的归属争议，涉及纳粹掠夺艺术品归还问题",
    1: "关于 Ferguson 事件中警察执法和 Mike Brown 枪击案的讨论",
    2: "关于 [待补充] 的讨论",
    3: "关于 [待补充] 的讨论",
    4: "关于 [待补充] 的讨论",
    5: "关于 [待补充] 的讨论",
    6: "关于 [待补充] 的讨论",
}
```

> ⚠️ 事件背景需要你们根据训练数据的推文内容来补充完善。

---

#### 成员 B 交付清单

```
retrieval.py             # 案例检索（CaseRetriever 类 + index 构建）
llm_explainer.py         # LLM 解释生成（LLMExplainer 类 + prompt 模板）
event_context.py         # 事件背景信息
data/
  index.pkl              # 训练集向量索引（预计算，加速推理）
.env.example             # API key 配置模板
```

**对外接口**（供成员 C 集成）:
```python
def explain(text: str, dl_result: dict) -> str:
    """
    Args:
        text: 原推文文本
        dl_result: 成员 A 的 predict() 返回结果
        
    Returns:
        自然语言解释文本（中文）
    """
```

---

### 👤 成员 C：系统集成 + 评估 + 报告

**职责**: 串联 A 和 B 的模块，构建端到端推理管道，全面评估系统性能，撰写报告。

#### 1. 系统集成

**目标**: 构建单脚本推理管道，一键运行。

```python
# inference.py — 主推理脚本
import argparse
import pandas as pd
from models.classifier import RumorClassifier
from models.keyword_extractor import extract_keywords
from retrieval import CaseRetriever
from llm_explainer import LLMExplainer
from event_context import EVENT_CONTEXT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 CSV 文件路径')
    parser.add_argument('--output', required=True, help='输出 CSV 文件路径')
    parser.add_argument('--model', default='checkpoints/best_model.pt')
    parser.add_argument('--no-llm', action='store_true', help='跳过 LLM 调用，仅输出分类结果')
    args = parser.parse_args()
    
    # 加载模型
    classifier = RumorClassifier()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()
    
    retriever = CaseRetriever()
    retriever.load_index('data/index.pkl')
    
    explainer = LLMExplainer()
    
    # 读取数据
    df = pd.read_csv(args.input)
    
    results = []
    for _, row in df.iterrows():
        text = row['text']
        event = int(row['event'])
        
        # Step 1: DL 分类
        dl_result = predict(classifier, tokenizer, text, event)
        
        # Step 2: 相似案例检索
        cases = retriever.search(text, top_k=3)
        
        # Step 3: LLM 解释
        if not args.no_llm:
            explanation = explainer.explain(
                text, dl_result, cases, EVENT_CONTEXT.get(event, "")
            )
        else:
            explanation = ""
        
        results.append({
            'id': row['id'],
            'text': text,
            'true_label': row['label'],
            'pred_label': dl_result['label'],
            'confidence': dl_result['confidence'],
            'keywords': ','.join([w for w, _ in dl_result['keywords']]),
            'explanation': explanation
        })
    
    # 保存结果
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"推理完成，结果保存至 {args.output}")
```

**运行命令**:
```bash
# 完整推理（含 LLM）
python inference.py --input rumer2026/val.csv --output results/val_results.csv

# 仅分类（快速测试，不调 LLM）
python inference.py --input rumer2026/val.csv --output results/val_results.csv --no-llm
```

**验收标准**:
- `inference.py` 可一键运行，输入 CSV → 输出含预测和解释的 CSV
- 无 GPU 时也能 CPU 运行（速度慢但不出错）
- `--no-llm` 模式可在 30 秒内完成

---

#### 2. 系统评估

**评估指标**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 整体评估
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 每个事件分别评估
for event_id in range(7):
    mask = (y_event == event_id)
    event_acc = accuracy_score(y_true[mask], y_pred[mask])
    event_f1 = f1_score(y_true[mask], y_pred[mask])
```

**需要产出的图表**:
1. 混淆矩阵（热力图）
2. 各事件准确率柱状图（对比整体均值）
3. 置信度分布直方图（正确预测 vs 错误预测）
4. 置信度校准曲线（reliability diagram）
5. 各事件 F1 对比图

> 使用 `matplotlib` + `seaborn` 绘制，图表需有中文标题和轴标签。

---

#### 3. 跨事件泛化评估（Leave-One-Event-Out）

见[优化方向六](#6️⃣-跨事件泛化评估)。

---

#### 4. 报告撰写

报告按模板结构组织：

1. **任务目标** — 可解释谣言检测
2. **具体内容**
   - (1) 实施方案 — 系统架构图 + 技术选型说明
   - (2) 核心代码分析 — 分类器、解释器、系统集成关键代码片段
   - (3) 检测结果分析 — 评估指标表 + 图表 + 错误案例分析
   - (4) 判断依据分析 — 解释质量评估 + 关键词有效性 + 置信度分级效果
3. **工作总结** — 收获心得 + 遇到的问题及解决
4. **课程建议**

---

#### 成员 C 交付清单

```
inference.py             # 端到端推理脚本
evaluate.py              # 评估脚本（指标 + 图表）
results/
  val_results.csv        # 验证集推理结果
  evaluation_report.txt  # 评估文本报告
figures/
  confusion_matrix.png
  event_accuracy.png
  confidence_hist.png
  calibration_curve.png
  event_f1_comparison.png
report/
  报告.docx 或 报告.pdf  # 最终报告
```

---

## Git 分支管理与协作规范

### 分支策略

```
main ───────────────────────────────────────────────→
  │
  ├── member-a/classifier ──→ PR → C review → merge
  ├── member-b/explainer  ──→ PR → C review → merge
  └── member-c/integration──→ PR → merge
```

### 命名规则

| 分支类型 | 格式 | 示例 |
|---------|------|------|
| 功能分支 | `member-{a/b/c}/{模块}` | `member-a/classifier`, `member-b/retrieval` |
| 修复分支 | `fix/{描述}` | `fix/api-timeout`, `fix/preprocess-url` |
| 实验分支 | `exp/{描述}` | `exp/data-augmentation`, `exp/leave-one-event` |

### 工作流

**1. 每人从 main 拉自己的分支**
```bash
git checkout main && git pull origin main
git checkout -b member-a/classifier   # A
git checkout -b member-b/explainer    # B
git checkout -b member-c/integration  # C
```

**2. 提交到自己的远程分支**
```bash
git add <files>
git commit -m "feat: xxx"
git push origin member-a/classifier
```

**3. 发起 Pull Request → C review → Merge**

### Commit 规范

```
feat:     新功能    feat: 添加 attention 关键词提取
fix:      修复      fix: URL 清洗正则遗漏 case
docs:     文档      docs: 更新 API 配置说明
refactor: 重构      refactor: 简化分类器前向传播
test:     测试      test: 添加跨事件评估测试
```

### 注意事项

- ⚠️ **永远不要在 main 分支上直接开发**
- ⚠️ **不要 force push 到 main**
- ⚠️ **不要提交 `.env` 和模型权重文件**（已在 `.gitignore`）
- ⚠️ **merge 前确保代码在自己机器上跑通**

> 详细协作流程、接口契约、集成 check list 见 **[DEVELOPMENT.md](DEVELOPMENT.md)**

---

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch==2.1.0
transformers==4.36.0
sentence-transformers==2.2.2
openai==1.6.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
tqdm==4.65.0
python-dotenv==1.0.0
```

### API 配置

复制 `.env.example` 为 `.env`，填入 SJTU API key：

```
SJTU_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**获取 API key**:
1. 登录 [my.sjtu.edu.cn](https://my.sjtu.edu.cn/)
2. 搜索「"致远一号"AI模型API申请（测试）」→ 点击流程申请大模型API
3. 申请通过后邮箱和交我办消息中收到 `base_url` 和 `api-key`

> ⚠️ **注意事项**：
> - API 仅限**校园网**访问（校外需通过 SJTU VPN）
> - 请求中**必须包含 `user` 角色的消息**，否则 DeepSeek V3.2 不返回内容
> - 速率限制：100 次/分钟，100,000 tokens/分钟，1,000,000,000 tokens/周
> - API key 有效期至 2026-06-30

---

## 项目目录结构

```
AI_rumor_detection/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── .env.example                  # API 配置模板
├── .gitignore
│
├── rumer2026/                    # 原始数据
│   ├── train.csv                 # 训练集 (2840条)
│   └── val.csv                   # 验证集 (401条)
│
├── data/                         # 预处理后的数据
│   ├── index.pkl                 # 检索索引（成员 B 生成）
│   ├── processed_train.csv       # 清洗后的训练数据
│   └── processed_val.csv         # 清洗后的验证数据
│
├── models/                       # DL 模型相关
│   ├── classifier.py             # 模型定义（成员 A）
│   └── keyword_extractor.py      # 关键词提取（成员 A）
│
├── checkpoints/                  # 训练产物
│   └── best_model.pt             # 最佳模型权重（成员 A 产出）
│
├── logs/                         # 训练日志
│   └── training.log
│
├── train.py                      # 训练脚本（成员 A）
├── preprocess.py                 # 数据预处理（成员 A）
│
├── retrieval.py                  # 案例检索（成员 B）
├── llm_explainer.py              # LLM 解释（成员 B）
├── event_context.py              # 事件背景（成员 B）
│
├── inference.py                  # 端到端推理（成员 C）
├── evaluate.py                   # 评估脚本（成员 C）
│
├── results/                      # 推理结果
│   └── val_results.csv
│
├── figures/                      # 评估图表
│   ├── confusion_matrix.png
│   ├── event_accuracy.png
│   ├── confidence_hist.png
│   ├── calibration_curve.png
│   └── event_f1_comparison.png
│
└── report/                       # 最终报告
    └── 报告.docx
```

---

## 可复现性保障

| 措施 | 说明 |
|------|------|
| `requirements.txt` | 精确版本号，`pip install -r requirements.txt` |
| 随机种子固定 | `seed=42`（PyTorch, NumPy, Python random） |
| 模型权重本地化 | `checkpoints/best_model.pt`，不依赖网络下载 |
| API key 环境变量 | `os.getenv("SJTU_API_KEY")`，不硬编码 |
| 单脚本推理 | `python inference.py --input xxx.csv --output xxx.csv` |
| GPU 非必需 | CPU 可运行（推理用 CPU，训练建议用 GPU 加速） |

### 固定随机种子的代码

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 未来优化方向

> 基础版本跑通后，按优先级逐步尝试。每个优化完成后记录效果对比。

### 1️⃣ 引入 Event 信息（优先级：⭐⭐⭐⭐⭐）

**纳入基础版**

**问题**: 不同事件的谣言比例差异巨大（Event 2 全谣言 vs Event 0 仅 20%），event 是强信号。

**做法**:
```python
# 在文本前拼接事件 token
text = f"[EVENT_{event_id}] {original_text}"
# BERT 能学到 event 和谣言之间的关联
```

**预期效果**: 准确率提升 2-3%
**工作量**: ~5 行代码

---

### 2️⃣ 置信度分级解释（优先级：⭐⭐⭐⭐⭐）

**纳入基础版**

**问题**: 置信度 0.51 和 0.95 都判谣言，但用户对两者的信任程度应完全不同。

**做法**:
```python
def confidence_level(conf):
    if conf > 0.9:    return "高度确信"
    elif conf > 0.7:  return "倾向于判定"
    else:             return "建议人工复核"
```

**预期效果**: 解释更诚实，展现系统对自身局限的认知
**工作量**: ~10 行代码

---

### 3️⃣ 跨事件泛化评估（优先级：⭐⭐⭐⭐）

**评估阶段做**

**问题**: 标准随机评估过于乐观，模型可能在事件间泛化差。

**做法**: Leave-One-Event-Out (LOEO)
```python
# 7 折，每次留 1 个 event 做测试
for held_out_event in range(7):
    train_data = df[df['event'] != held_out_event]
    test_data = df[df['event'] == held_out_event]
    # 训练 → 评估 → 记录
```

**报告产出**:
- 7 折准确率表 + 均值
- 各事件难度分析
- 极端事件（Event 2）的表现分析

**预期效果**: 报告中有深度的泛化分析
**工作量**: ~50 行代码

---

### 4️⃣ 解释质量人工自评（优先级：⭐⭐⭐⭐）

**评估阶段做**

**做法**: 三人对 50 条 LLM 解释进行三维度打分

| 维度 | 标准 | 1分 | 3分 | 5分 |
|------|------|-----|-----|-----|
| 忠实性 | 解释是否与 DL 模型关注的词一致 | 完全不相关 | 部分相关 | 高度一致 |
| 可读性 | 普通人能否理解 | 难以理解 | 基本通顺 | 清晰流畅 |
| 信息量 | 是否有具体线索而非泛泛而谈 | 全是套话 | 有部分线索 | 具体有据 |

**报告产出**:
- 评分分布图
- 高低分案例对比
- 解释质量改进方向

**工作量**: 2-3 小时（三人各自评分 50 条 + 汇总分析）

---

### 5️⃣ 短文本数据增强（优先级：⭐⭐⭐）

**时间充裕时做**

**问题**: 平均 16.3 词，模型容易记忆关键词而非学习语义模式。

**方法**:
- **回译增强**: 英文 → 中文 → 回译英文，保持语义但换表达
  ```python
  # 示例
  原文: "Police fired tear gas at protesters"
  中译: "警方对抗议者发射催泪瓦斯"
  回译: "Police launched tear gas at demonstrators"
  ```
- **同义词替换**: 用 WordNet 随机替换非核心名词/动词
- **随机删除**: 以概率 p 随机删除非关键词

**注意**: 增强后需保持标签不变，增强比例控制在 50% 以内。

**预期效果**: 泛化能力提升，尤其对稀有事件
**工作量**: ~100 行代码 + 额外训练时间

---

### 6️⃣ 混合检索优化（优先级：⭐⭐⭐）

**时间充裕时做**

**问题**: 16 词推文纯语义检索效果不稳定。

**改进方案**:
- **语义 + 关键词混合打分**:
  ```python
  final_score = 0.7 * semantic_similarity + 0.3 * keyword_overlap
  ```
- **事件内聚类代表推文**: 对每个事件的推文先聚类，检索时返回各簇的代表案例
- **确保标签多样性**: 检索结果中强制包含至少 1 条谣言和 1 条非谣言

**预期效果**: 检索质量更稳定，LLM 引用时参考价值更高
**工作量**: ~150 行代码

---

### 优化优先级总览

```
必须做（基础版包含）:
  1️⃣ Event 信息引入      ★★★★★
  2️⃣ 置信度分级解释        ★★★★★

评估阶段做:
  3️⃣ 跨事件泛化评估        ★★★★
  4️⃣ 解释质量自评          ★★★★

时间充裕再做:
  5️⃣ 短文本数据增强        ★★★
  6️⃣ 混合检索优化          ★★★
```

---

## 开发时间线建议

| 阶段 | 内容 | 建议时间 |
|------|------|----------|
| 第1周 | 环境搭建、数据预处理（A）、检索索引构建（B）、集成框架（C） | 并行开发 |
| 第2周 | 分类器训练+调参（A）、提示词调试+API测试（B）、评估脚本（C） | 并行开发 |
| 第3周 | 模块联调、端到端测试、优化项 1️⃣ 2️⃣ | 集成测试 |
| 第4周 | 优化项 3️⃣ 4️⃣、报告撰写（C 主笔，A B 补充各自部分） | 收尾 |

---

## 常见问题

**Q: API 调用失败怎么办？**
A: 检查 `.env` 文件中的 `SJTU_API_KEY` 是否正确，网络是否可达 `api.claw.sjtu.edu.cn`。使用 `--no-llm` 模式可以跳过 LLM 调用测试分类器部分。

**Q: GPU 内存不够？**
A: 减小 batch_size 到 8 或 4，或使用 `--fp16` 混合精度训练。

**Q: 模型准确率不达标？**
A: 检查预处理是否正确（URL 是否去除干净），尝试调整学习率和 dropout 参数，确保 event token 正确拼接。

**Q: 解释质量不好？**
A: 调整 prompt 模板，增加更具体的约束；提高相似案例质量；降低 LLM temperature 参数。

---

## 参考资料

### SJTU API 可用模型

| 模型名称 | 参数 | 调用 ID | 上下文 | 适用场景 |
|---------|------|---------|--------|---------|
| DeepSeek V3.2 | 685B | `deepseek-chat` | 32k | **通用文本（推荐）** |
| DeepSeek V3.2 Think | 685B | `deepseek-reasoner` | 32k | 复杂推理 |
| MiniMax-M2.7 | 230B | `minimax` / `minimax-m2.7` | 192k | 智能体任务 |
| GLM-5.1 | 754B | `glm` / `glm-5.1` | 128k | 代码与长程任务 |
| Qwen3.5-27B | 27B | `qwen` / `qwen3.5-27b` | 256k | 视觉+文本理解 |

**API 端点**: `https://models.sjtu.edu.cn/api/v1/chat/completions`（OpenAI 兼容格式）
**认证方式**: `Authorization: Bearer <your-api-key>`

### 外部链接

- SJTU API 文档: https://claw.sjtu.edu.cn/guide/sjtu-api/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Sentence Transformers: https://www.sbert.net/
