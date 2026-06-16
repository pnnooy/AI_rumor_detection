"""
训练脚本 — 姜新晨 实现

训练 BERT 谣言分类器，包含:
  - 标准训练模式
  - 对抗训练模式（--adversarial，用于优化 7️⃣）
  - 每 epoch 验证 + 最佳模型保存
  - 训练日志记录

用法:
    # 标准训练
    python train.py

    # 自定义超参数
    python train.py --epochs 10 --batch_size 32 --lr 3e-5

    # 对抗训练（配合 optimization 7️⃣）
    python train.py --adversarial --epochs 8

    # 完整配置
    python train.py \
        --train_csv rumer2026/train.csv \
        --val_csv rumer2026/val.csv \
        --epochs 5 \
        --batch_size 16 \
        --lr 2e-5 \
        --max_len 64 \
        --save_dir checkpoints \
        --log_dir logs \
        --seed 42
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from models.classifier import RumorClassifier
from preprocess import create_dataloaders, set_seed


# ============================================================
# 学习率调度器（Linear Warmup + Linear Decay）
# ============================================================

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    创建带 warmup 的线性学习率调度器

    前 num_warmup_steps 步从 0 线性增长到 lr，
    之后线性衰减到 0。
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 训练函数
# ============================================================

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int,
    adversarial: bool = False,
    adv_weight: float = 0.5,
) -> dict:
    """
    训练一个 epoch

    Args:
        model:        分类模型
        dataloader:   训练 DataLoader
        optimizer:    优化器
        scheduler:    学习率调度器
        criterion:    损失函数
        device:       设备
        epoch:        当前 epoch
        total_epochs: 总 epoch 数
        adversarial:  是否开启对抗训练
        adv_weight:   对抗损失权重

    Returns:
        {'loss': float, 'accuracy': float}
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # === 标准训练 ===
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, output_attentions=False)
        loss_clean = criterion(logits, labels)

        # === 对抗训练 ===
        if adversarial and step % 5 == 0:
            # 每 5 个 batch 做一次对抗训练
            try:
                from adversarial import generate_adversarial

                # 生成对抗样本（需要原文本）
                # 从 batch 中还原文本
                adv_texts = []
                for i in range(input_ids.size(0)):
                    # 尝试从 input_ids 还原原始事件 ID
                    event_id = batch.get('event', [0] * input_ids.size(0))
                    eid = event_id[i] if isinstance(event_id, (list, torch.Tensor)) else 0
                    if isinstance(eid, torch.Tensor):
                        eid = eid.item()

                    # 用解码+扰动的方式生成对抗文本
                    ids = input_ids[i].cpu().tolist()
                    # 去掉 event token 再解码
                    decoded = tokenizer_global.decode(ids, skip_special_tokens=True)
                    adv_text = generate_adversarial(decoded, max_swaps=2)
                    adv_texts.append(f"[EVENT_{eid}] {adv_text}")

                adv_inputs = tokenizer_global(
                    adv_texts, padding='max_length', truncation=True,
                    max_length=64, return_tensors='pt'
                )
                adv_logits = model(
                    adv_inputs['input_ids'].to(device),
                    adv_inputs['attention_mask'].to(device),
                    output_attentions=False
                )
                loss_adv = criterion(adv_logits, labels)
                loss = loss_clean + adv_weight * loss_adv
            except ImportError:
                loss = loss_clean
            except Exception:
                loss = loss_clean
        else:
            loss = loss_clean

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    return {'loss': avg_loss, 'accuracy': acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str
) -> dict:
    """
    验证集评估

    Returns:
        {'loss': float, 'accuracy': float, 'precision': float,
         'recall': float, 'f1': float}
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="[Val]", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask, output_attentions=False)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


# ============================================================
# 训练主流程
# ============================================================

def run_training(args):
    """完整训练流程"""
    set_seed(args.seed)

    # --- 创建输出目录 ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- 日志设置 ---
    log_file = os.path.join(args.log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("BERT 谣言分类器 — 训练开始")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # --- 设备 ---
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"设备: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 超参数 ---
    logger.info(f"\n超参数:")
    for key in ['epochs', 'batch_size', 'lr', 'max_len', 'dropout',
                'warmup_ratio', 'weight_decay', 'seed', 'adversarial']:
        logger.info(f"  {key}: {getattr(args, key)}")

    # --- 数据 ---
    logger.info(f"\n加载数据...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        tokenizer_name=args.bert_model,
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # 保存 tokenizer 引用供对抗训练使用
    global tokenizer_global
    tokenizer_global = tokenizer

    # --- 模型 ---
    logger.info(f"\n创建模型...")
    model = RumorClassifier(num_classes=2, dropout=args.dropout)
    # 扩展 embedding 以容纳 event tokens
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

    # --- 优化器 & 调度器 ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"\n总训练步数: {total_steps}, warmup 步数: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    logger.info(f"\n{'='*60}")
    logger.info("开始训练")
    logger.info(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs,
            adversarial=args.adversarial, adv_weight=args.adv_weight
        )

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # 记录
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
        })

        # 打印 epoch 结果
        logger.info(
            f"\nEpoch {epoch}/{args.epochs} | 时间: {epoch_time:.1f}s\n"
            f"  Train  | loss: {train_metrics['loss']:.4f} | acc: {train_metrics['accuracy']:.4f}\n"
            f"  Val    | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f} "
            f"| prec: {val_metrics['precision']:.4f} | rec: {val_metrics['recall']:.4f} "
            f"| f1: {val_metrics['f1']:.4f}"
            + (' *BEST*' if val_metrics['accuracy'] > best_val_acc else '')
        )

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'num_classes': 2,
                'dropout': args.dropout,
                'max_len': args.max_len,
                'history': history,
                'timestamp': datetime.now().isoformat(),
            }, checkpoint_path)
            logger.info(f"  → 保存最佳模型: {checkpoint_path} (val_acc={best_val_acc:.4f})")

        # 早停（可选）
        if args.early_stop > 0 and epoch - best_epoch >= args.early_stop:
            logger.info(f"\n早停: {args.early_stop} 个 epoch 验证准确率未提升")
            break

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成 | 总时间: {total_time/60:.1f} 分钟")
    logger.info(f"最佳模型: epoch {best_epoch}, val_acc = {best_val_acc:.4f}")
    logger.info(f"{'='*60}")

    # --- 最终评估 ---
    logger.info(f"\n{'='*60}")
    logger.info("加载最佳模型进行最终评估")
    logger.info(f"{'='*60}")

    checkpoint = torch.load(
        os.path.join(args.save_dir, 'best_model.pt'),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate(model, val_loader, criterion, device)
    logger.info(f"\n最终验证集指标:")
    logger.info(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {final_metrics['recall']:.4f}")
    logger.info(f"  F1:        {final_metrics['f1']:.4f}")

    # --- 训练历史摘要 ---
    logger.info(f"\n训练历史:")
    logger.info(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'Val F1':>7}")
    logger.info(f"{'-'*60}")
    for h in history:
        logger.info(
            f"{h['epoch']:>6} {h['train_loss']:>11.4f} {h['train_acc']:>10.4f} "
            f"{h['val_loss']:>9.4f} {h['val_acc']:>8.4f} {h['val_f1']:>7.4f}"
        )

    logger.info(f"\n训练日志已保存至: {log_file}")
    logger.info(f"最佳模型已保存至: {os.path.join(args.save_dir, 'best_model.pt')}")

    return model, tokenizer, history


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT 谣言分类器 — 训练脚本"
    )

    # 数据
    parser.add_argument('--train_csv', default='rumer2026/train.csv',
                        help='训练集 CSV 路径')
    parser.add_argument('--val_csv', default='rumer2026/val.csv',
                        help='验证集 CSV 路径')
    parser.add_argument('--save_dir', default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', default='logs',
                        help='日志保存目录')

    # 模型
    parser.add_argument('--bert_model', default='bert-base-uncased',
                        help='预训练 BERT 模型名称')
    parser.add_argument('--max_len', type=int, default=64,
                        help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 概率')

    # 训练
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch 大小')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW 权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup 占比')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='早停 patience (0 = 不使用)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 设备
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='运行设备')

    # 对抗训练
    parser.add_argument('--adversarial', action='store_true',
                        help='开启对抗训练模式（优化 7️⃣）')
    parser.add_argument('--adv_weight', type=float, default=0.5,
                        help='对抗损失权重')

    return parser.parse_args()


# --- 全局 tokenizer 引用（对抗训练需要）---
tokenizer_global = None


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    args = parse_args()
    run_training(args)
