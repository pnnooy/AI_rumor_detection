"""
对抗训练 V2 — 多攻击混合 + 高强度参数

攻击类型（随机选一）:
  1. WordNet 同义词替换 (max_swaps=3)
  2. 随机删词 (1个非停用词)
  3. 随机交换相邻字符 (1处)

改进:
  - adv_weight = 1.0
  - 每 2 步注入对抗样本
  - 三种攻击随机混合

用法:
  python train_adv_v2.py --save_dir checkpoints/adv_v2 --device cuda
"""

import os, sys, time, random, logging, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from models.classifier import RumorClassifier
from preprocess import create_dataloaders, set_seed

# ============================================================
# 多攻击生成器
# ============================================================

# 停用词集合（常见英文功能词）
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
    'if', 'while', 'that', 'this', 'it', 'its', 'he', 'she', 'they',
    'them', 'their', 'his', 'her', 'my', 'your', 'our', 'we', 'you',
    'me', 'us', 'who', 'which', 'what', 'about', 'up', 'down', 'also',
    'am', 's', 't', 'don', 'll', 've', 're', 'm', 'didn', 'won',
}


def attack_synonym(text: str, max_swaps: int = 3, seed: int = None) -> str:
    """攻击1: WordNet 同义词替换"""
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return text

    if seed is not None:
        random.seed(seed)

    words = text.split()
    if len(words) < 4:
        return text

    # 找可替换词
    candidates = []
    for i, w in enumerate(words):
        clean = w.strip('#@.,!?;:\"\'()[]{}')
        if len(clean) < 3 or clean.lower() in STOP_WORDS:
            continue
        syns = wordnet.synsets(clean)
        if syns:
            lemmas = [l.name().replace('_', ' ') for s in syns
                      for l in s.lemmas() if l.name().lower() != clean.lower()]
            if lemmas:
                candidates.append((i, random.choice(lemmas)))

    if not candidates:
        return text

    swaps = min(max_swaps, len(candidates))
    for i, syn in random.sample(candidates, swaps):
        words[i] = syn

    return ' '.join(words)


def attack_delete(text: str, seed: int = None) -> str:
    """攻击2: 随机删一个非停用词"""
    if seed is not None:
        random.seed(seed)

    words = text.split()
    candidates = [i for i, w in enumerate(words)
                  if w.strip('#@.,!?;:\"\'()[]{}').lower() not in STOP_WORDS
                  and len(w.strip('#@.,!?;:\"\'()[]{}')) >= 3]

    if not candidates:
        return text

    i = random.choice(candidates)
    words.pop(i)
    return ' '.join(words)


def attack_charswap(text: str, seed: int = None) -> str:
    """攻击3: 随机交换相邻两个字符（在某个长度>=4的词内部）"""
    if seed is not None:
        random.seed(seed)

    words = text.split()
    candidates = [(i, j) for i, w in enumerate(words)
                  for j in range(len(w) - 1)
                  if len(w) >= 4 and w[j].isalpha() and w[j+1].isalpha()
                  and w[j] != w[j+1] and not w.startswith('#') and not w.startswith('@')]

    if not candidates:
        return text

    i, j = random.choice(candidates)
    w = list(words[i])
    w[j], w[j+1] = w[j+1], w[j]
    words[i] = ''.join(w)
    return ' '.join(words)


def generate_adv_v2(text: str, seed: int = None) -> str:
    """随机选择一种攻击"""
    attacks = [attack_synonym, attack_delete, attack_charswap]
    attack_fn = random.choice(attacks)
    if attack_fn == attack_synonym:
        return attack_fn(text, max_swaps=3, seed=seed)
    else:
        return attack_fn(text, seed=seed)


# ============================================================
# 训练工具
# ============================================================

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                epoch, total_epochs, tokenizer, max_len,
                adv_every_n_steps=2, adv_weight=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, output_attentions=False)
        loss = criterion(logits, labels)

        # ============================================================
        # 对抗训练: 每 N 步注入，三种攻击随机混合
        # ============================================================
        if step % adv_every_n_steps == 0:
            try:
                adv_texts = []
                for i in range(input_ids.size(0)):
                    ids = input_ids[i].cpu().tolist()
                    event_id = batch.get('event', [0] * input_ids.size(0))
                    eid = event_id[i] if isinstance(event_id, (list, torch.Tensor)) else 0
                    if isinstance(eid, torch.Tensor):
                        eid = eid.item()
                    decoded = tokenizer.decode(ids, skip_special_tokens=True)
                    adv_text = generate_adv_v2(decoded, seed=None)
                    adv_texts.append(f"[EVENT_{eid}] {adv_text}")

                adv_inputs = tokenizer(
                    adv_texts, padding='max_length', truncation=True,
                    max_length=max_len, return_tensors='pt'
                )
                adv_logits = model(
                    adv_inputs['input_ids'].to(device),
                    adv_inputs['attention_mask'].to(device),
                    output_attentions=False
                )
                loss = loss + adv_weight * criterion(adv_logits, labels)
            except Exception:
                pass

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

    return {'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds)}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

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
    return {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }


# ============================================================
# 主训练流程
# ============================================================

def run_training(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_file = os.path.join(args.log_dir, 'training_v2.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                  logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"对抗训练 V2 — 多攻击混合 — model={args.bert_model}")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"设备: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"\n超参数:")
    for key in ['epochs', 'batch_size', 'lr', 'max_len', 'dropout',
                'adv_weight', 'adv_every_n_steps', 'bert_model']:
        logger.info(f"  {key}: {getattr(args, key)}")
    logger.info(f"  攻击类型: WordNet同义词(3) / 随机删词 / 字符交换")
    logger.info(f"  攻击选择: 随机混合")

    logger.info(f"\n加载数据...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_csv=args.train_csv, val_csv=args.val_csv,
        tokenizer_name=args.bert_model,
        max_len=args.max_len, batch_size=args.batch_size, seed=args.seed
    )

    logger.info(f"\n创建模型 ({args.bert_model})...")
    model = RumorClassifier(
        model_name=args.bert_model,
        num_classes=2,
        dropout=args.dropout
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"\n总训练步数: {total_steps}, warmup: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    logger.info(f"\n{'='*60}")
    logger.info("开始训练")
    logger.info(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, tokenizer, args.max_len,
            adv_every_n_steps=args.adv_every_n_steps,
            adv_weight=args.adv_weight
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

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

        logger.info(
            f"\nEpoch {epoch}/{args.epochs} | 时间: {epoch_time:.1f}s\n"
            f"  Train  | loss: {train_metrics['loss']:.4f} | acc: {train_metrics['accuracy']:.4f}\n"
            f"  Val    | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f} "
            f"| prec: {val_metrics['precision']:.4f} | rec: {val_metrics['recall']:.4f} "
            f"| f1: {val_metrics['f1']:.4f}"
            + (' *BEST*' if val_metrics['accuracy'] > best_val_acc else '')
        )

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
                'model_name': args.bert_model,
                'max_len': args.max_len,
                'history': history,
                'timestamp': datetime.now().isoformat(),
                'adv_training': 'v2_multi_attack',
            }, checkpoint_path)
            logger.info(f"  → 保存最佳模型: {checkpoint_path} (val_acc={best_val_acc:.4f})")

        if args.early_stop > 0 and epoch - best_epoch >= args.early_stop:
            logger.info(f"\n早停: {args.early_stop} epoch 未提升")
            break

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成 | 总时间: {total_time/60:.1f} 分钟")
    logger.info(f"最佳模型: epoch {best_epoch}, val_acc = {best_val_acc:.4f}")
    logger.info(f"{'='*60}")

    # 最终评估
    logger.info(f"\n{'='*60}")
    logger.info("加载最佳模型进行最终评估")
    logger.info(f"{'='*60}")

    checkpoint = torch.load(
        os.path.join(args.save_dir, 'best_model.pt'),
        map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate(model, val_loader, criterion, device)
    logger.info(f"\n最终验证集指标:")
    logger.info(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {final_metrics['recall']:.4f}")
    logger.info(f"  F1:        {final_metrics['f1']:.4f}")

    logger.info(f"\n训练历史:")
    logger.info(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'Val F1':>7}")
    logger.info(f"{'-'*60}")
    for h in history:
        logger.info(f"{h['epoch']:>6} {h['train_loss']:>11.4f} {h['train_acc']:>10.4f} "
                    f"{h['val_loss']:>9.4f} {h['val_acc']:>8.4f} {h['val_f1']:>7.4f}")

    logger.info(f"\n训练日志: {log_file}")
    logger.info(f"最佳模型: {os.path.join(args.save_dir, 'best_model.pt')}")
    return model, tokenizer, history


def parse_args():
    parser = argparse.ArgumentParser(description="对抗训练 V2 — 多攻击混合")
    parser.add_argument('--train_csv', default='rumer2026/train.csv')
    parser.add_argument('--val_csv', default='rumer2026/val.csv')
    parser.add_argument('--save_dir', default='checkpoints/adv_v2')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--bert_model', default='roberta-large')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--adv_weight', type=float, default=1.0)
    parser.add_argument('--adv_every_n_steps', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
