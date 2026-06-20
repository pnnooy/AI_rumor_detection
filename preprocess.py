"""
数据预处理 — 姜新晨 实现

包含:
  1. clean_text() — 文本清洗
  2. RumorDataset — PyTorch Dataset
  3. set_seed() — 可复现性
  4. create_dataloaders() — DataLoader 工厂函数

用法:
    from preprocess import RumorDataset, create_dataloaders, set_seed, clean_text

    set_seed(42)
    train_loader, val_loader, tokenizer = create_dataloaders(
        "rumer2026/train.csv",
        "rumer2026/val.csv",
        batch_size=16
    )
"""

import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# ============================================================
# 可复现性
# ============================================================

def set_seed(seed: int = 42):
    """固定所有随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """
    清洗单条推文文本

    处理步骤:
        1. 去除 URL (http://t.co/..., https://...)
        2. 去除 HTML 实体 (&amp;, &lt;, &gt; 等)
        3. 保留 hashtag 文本（去掉 # 符号保留词）
        4. @ 提及统一替换为 @USER
        5. 去除多余空格、首尾空格

    Args:
        text: 原始推文文本

    Returns:
        清洗后的文本

    Examples:
        >>> clean_text("See http://t.co/abc #Ferguson news")
        "See Ferguson news"
        >>> clean_text("@police said &amp; done")
        "@USER said & done"
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. 去除 URL
    text = re.sub(r'http\S+', '', text)

    # 2. 处理常见 HTML 实体
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&\w+;', ' ', text)

    # 3. 保留 hashtag 词（去掉 # 符号）
    text = re.sub(r'#(\w+)', r'\1', text)

    # 4. @ 提及统一替换为 @USER
    text = re.sub(r'@\w+', '@USER', text)

    # 5. 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# Dataset
# ============================================================

class RumorDataset(Dataset):
    """
    谣言检测 PyTorch Dataset

    对每条推文:
        1. 清洗文本
        2. 拼接事件 token: [EVENT_{event_id}] + 清洗后的文本
        3. BERT tokenization → input_ids, attention_mask

    Args:
        csv_path:   CSV 文件路径 (含 columns: id, text, label, event)
        tokenizer:  HuggingFace BERT tokenizer
        max_len:    最大序列长度 (默认 64)
        is_train:   是否训练模式
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BertTokenizer,
        max_len: int = 64,
        is_train: bool = True
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        # 验证必要列存在
        required_cols = ['text', 'label', 'event']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV 缺少必需列: {col} (文件: {csv_path})")

        # 打印数据集信息
        n_total = len(self.df)
        n_rumor = int(self.df['label'].sum())
        print(f"  加载数据集: {csv_path}")
        print(f"    样本数: {n_total}, 谣言: {n_rumor} ({n_rumor/n_total*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = clean_text(str(row['text']))
        event = int(row['event'])
        label = int(row['label'])

        # 拼接 event token 到文本开头
        text_with_event = f"[EVENT_{event}] {text}"

        # Tokenization
        encoding = self.tokenizer(
            text_with_event,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'event': event,
        }


# ============================================================
# DataLoader 工厂函数
# ============================================================

def create_dataloaders(
    train_csv: str = "rumer2026/train.csv",
    val_csv: str = "rumer2026/val.csv",
    tokenizer_name: str = "bert-base-uncased",
    max_len: int = 64,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, BertTokenizer]:
    """
    创建训练集和验证集的 DataLoader

    Args:
        train_csv:      训练集 CSV 路径
        val_csv:        验证集 CSV 路径
        tokenizer_name: HuggingFace tokenizer 名称
        max_len:        最大序列长度
        batch_size:     batch 大小
        num_workers:    DataLoader 工作进程数
        seed:           随机种子

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    set_seed(seed)

    print("加载 BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # 添加 event special tokens
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)
    print(f"  已添加 event tokens: {event_tokens}")

    print("\n创建 Dataset...")
    train_dataset = RumorDataset(train_csv, tokenizer, max_len=max_len, is_train=True)
    val_dataset = RumorDataset(val_csv, tokenizer, max_len=max_len, is_train=False)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nDataLoader 创建完成:")
    print(f"  训练集: {len(train_dataset)} 条, {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  验证集: {len(val_dataset)} 条, {len(val_loader)} batches")

    return train_loader, val_loader, tokenizer


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("数据预处理模块自测")
    print("=" * 60)

    # 1. 文本清洗测试
    print("\n[1] 文本清洗测试:")
    test_cases = [
        ("See http://t.co/abc #Ferguson news", "See Ferguson news"),
        ("@police said &amp; done", "@USER said & done"),
        ("   extra   spaces   ", "extra spaces"),
        ("#BREAKING - Museum to accept #Gurlitt art", "BREAKING - Museum to accept Gurlitt art"),
    ]
    for raw, expected in test_cases:
        result = clean_text(raw)
        ok = "OK" if result == expected else "FAIL"
        print(f"  {ok} '{raw[:55]}' -> '{result}'")

    # 2. Dataset 加载测试
    print("\n[2] Dataset 加载测试:")
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)

    try:
        dataset = RumorDataset("rumer2026/val.csv", tokenizer, max_len=64)
        sample = dataset[0]
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  label: {sample['label']}, event: {sample['event']}")
        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)[:100]
        print(f"  解码: {decoded}")
        print(f"  [OK] Dataset 加载成功")
    except Exception as e:
        print(f"  [FAIL] 加载失败: {e}")

    # 3. DataLoader 测试
    print("\n[3] DataLoader 测试:")
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        print(f"  batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  batch labels: {batch['label'].tolist()}")
        print(f"  [OK] DataLoader 正常")
    except Exception as e:
        print(f"  [FAIL] DataLoader 失败: {e}")

    print("\n" + "=" * 60)
    print("自测完成")
