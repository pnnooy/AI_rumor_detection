"""
数据预处理 — AutoModel 通用版
支持: bert-base-uncased, roberta-base, roberta-large, etc.
"""

import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RumorDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int = 64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        required_cols = ['text', 'label', 'event']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV 缺少必需列: {col} (文件: {csv_path})")
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
        text_with_event = f"[EVENT_{event}] {text}"
        encoding = self.tokenizer(
            text_with_event, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'event': event,
        }


def create_dataloaders(
    train_csv: str = "rumer2026/train.csv",
    val_csv: str = "rumer2026/val.csv",
    tokenizer_name: str = "bert-base-uncased",
    max_len: int = 64,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
):
    set_seed(seed)
    print(f"加载 tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)
    print(f"  已添加 event tokens, vocab 大小: {len(tokenizer)}")

    print("\n创建 Dataset...")
    train_dataset = RumorDataset(train_csv, tokenizer, max_len=max_len)
    val_dataset = RumorDataset(val_csv, tokenizer, max_len=max_len)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"\nDataLoader 创建完成: 训练 {len(train_dataset)}条/{len(train_loader)}batch, 验证 {len(val_dataset)}条/{len(val_loader)}batch")
    return train_loader, val_loader, tokenizer
