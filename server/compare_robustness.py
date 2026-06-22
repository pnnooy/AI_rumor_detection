"""
鲁棒性对比：评估所有 checkpoint 在三种攻击下的表现

用法:
  python compare_robustness.py
  python compare_robustness.py --n-samples 200
"""

import os, sys, random, argparse, time
from collections import Counter

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 清代理
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

from models.classifier import RumorClassifier
from preprocess import clean_text
from transformers import AutoTokenizer

# ============================================================
# 三种攻击（对齐 train_adv_v2.py）
# ============================================================

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


def attack_synonym(text: str, max_swaps: int = 3) -> str:
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return text
    words = text.split()
    if len(words) < 4:
        return text
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


def attack_delete(text: str) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words)
                  if w.strip('#@.,!?;:\"\'()[]{}').lower() not in STOP_WORDS
                  and len(w.strip('#@.,!?;:\"\'()[]{}')) >= 3]
    if not candidates:
        return text
    words.pop(random.choice(candidates))
    return ' '.join(words)


def attack_charswap(text: str) -> str:
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


ATTACKS = {
    '同义词替换': attack_synonym,
    '随机删词': attack_delete,
    '字符交换': attack_charswap,
}


# ============================================================
# 模型评估
# ============================================================

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', None)
    if model_name is None:
        w = ckpt['model_state_dict']['classifier.0.weight']
        model_name = 'roberta-large' if w.shape[1] == 1024 else 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    event_tokens = [f'[EVENT_{i}]' for i in range(7)]
    tokenizer.add_tokens(event_tokens)

    model = RumorClassifier(model_name=model_name, num_classes=2,
                            dropout=ckpt.get('dropout', 0.2))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, model_name, ckpt.get('val_accuracy', 0)


def evaluate_robustness(model, tokenizer, df, attack_name, attack_fn, device, seed=42):
    """测试模型在特定攻击下的翻转率"""
    random.seed(seed)
    np.random.seed(seed)

    flipped, total, high_conf_flipped = 0, 0, 0
    direction_0to1, direction_1to0 = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {attack_name}", leave=False):
        text = clean_text(str(row['text']))
        event_id = int(row['event'])

        # 原始预测
        inputs = tokenizer(f"[EVENT_{event_id}] {text}", return_tensors='pt',
                          max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            logits = model(inputs['input_ids'].to(device),
                          inputs['attention_mask'].to(device))
            prob = torch.softmax(logits, dim=-1)
            orig_conf = prob.max().item()
            orig_label = logits.argmax(dim=-1).item()

        # 攻击
        adv_text = attack_fn(text)
        if adv_text == text:
            continue

        adv_inputs = tokenizer(f"[EVENT_{event_id}] {adv_text}", return_tensors='pt',
                              max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            adv_logits = model(adv_inputs['input_ids'].to(device),
                              adv_inputs['attention_mask'].to(device))
            adv_label = adv_logits.argmax(dim=-1).item()

        total += 1
        if orig_label != adv_label:
            flipped += 1
            if orig_conf > 0.9:
                high_conf_flipped += 1
            if orig_label == 0 and adv_label == 1:
                direction_0to1 += 1
            elif orig_label == 1 and adv_label == 0:
                direction_1to0 += 1

    return {
        'total': total,
        'flipped': flipped,
        'flip_rate': flipped / total * 100 if total > 0 else 0,
        'high_conf_flipped': high_conf_flipped,
        'direction_0to1': direction_0to1,
        'direction_1to0': direction_1to0,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="鲁棒性对比评估")
    parser.add_argument('--n-samples', type=int, default=200,
                        help='测试样本数 (默认200，0=全部401)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 要对比的模型
    models_to_test = []
    for path, label in [
        ('checkpoints/best_model.pt', '原始 (BERT-base)'),
        ('checkpoints/adv_defense/best_model.pt', '对抗V1 (同义词)'),
        ('checkpoints/adv_v2/best_model.pt', '对抗V2 (多攻击)'),
    ]:
        if os.path.exists(path):
            models_to_test.append((path, label))
        else:
            print(f"[SKIP] 未找到: {path}")

    if not models_to_test:
        print("未找到任何模型！")
        return

    # 加载数据
    df = pd.read_csv('rumer2026/val.csv')
    if args.n_samples > 0 and args.n_samples < len(df):
        df = df.head(args.n_samples)
    print(f"\n测试样本: {len(df)} 条")
    print(f"攻击类型: {list(ATTACKS.keys())}")
    print(f"对比模型: {len(models_to_test)} 个\n")

    # 逐模型、逐攻击测试
    all_results = {}

    for path, label in models_to_test:
        print(f"{'='*60}")
        print(f"加载: {label}")
        model, tokenizer, model_name, val_acc = load_model(path, device)
        print(f"  架构: {model_name}, val_acc={val_acc:.4f}")

        model_results = {}
        for attack_name, attack_fn in ATTACKS.items():
            result = evaluate_robustness(model, tokenizer, df, attack_name, attack_fn, device)
            model_results[attack_name] = result
            print(f"  {attack_name}: 翻转 {result['flipped']}/{result['total']} "
                  f"({result['flip_rate']:.1f}%) "
                  f"高置信={result['high_conf_flipped']} "
                  f"0→1={result['direction_0to1']} 1→0={result['direction_1to0']}")

        all_results[label] = {
            'val_acc': val_acc,
            'model_name': model_name,
            'attacks': model_results,
        }

        # 释放显存
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # ============================================================
    # 汇总表
    # ============================================================
    print(f"\n{'='*80}")
    print("鲁棒性对比汇总")
    print(f"{'='*80}")

    attack_names = list(ATTACKS.keys())

    # 表头
    header = f"{'模型':<22} {'架构':<16} {'干净Acc':>8}"
    for a in attack_names:
        header += f" {a:>10}"
    header += f" {'平均翻转':>10} {'高置信翻':>8}"
    print(header)
    print('-' * len(header))

    for label, data in all_results.items():
        row = f"{label:<22} {data['model_name']:<16} {data['val_acc']:>7.4f}"
        avg_flip = 0
        avg_high = 0
        for a in attack_names:
            r = data['attacks'][a]
            row += f" {r['flip_rate']:>9.1f}%"
            avg_flip += r['flip_rate']
            avg_high += r['high_conf_flipped']
        avg_flip /= len(attack_names)
        row += f" {avg_flip:>9.1f}% {avg_high:>8}"
        print(row)

    print('-' * len(header))

    # 最佳模型标注
    if len(all_results) >= 2:
        labels = list(all_results.keys())
        best_flip = min(all_results[l]['attacks'][attack_names[0]]['flip_rate'] for l in labels)
        print(f"\n同义词攻击最低翻转率: {best_flip:.1f}%")

    print(f"\n完成! 测试 {len(df)} 条 × {len(ATTACKS)} 种攻击 × {len(models_to_test)} 个模型")


if __name__ == '__main__':
    main()
