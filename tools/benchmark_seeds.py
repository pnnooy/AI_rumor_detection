"""
多种子鲁棒性测试 — 看翻转率波动范围

用法:
    python benchmark_seeds.py              # 5个种子
    python benchmark_seeds.py --seeds 5    # 5个随机种子
"""

import random
import numpy as np
import pandas as pd
import os, sys, time

from models.classifier import load_model as load_dl
from models.keyword_extractor import predict, _init_predictor
from adversarial import generate_adversarial

SEEDS = [42, 123, 456, 789, 1024]
MODELS = [
    ("checkpoints/model_clean.pt", "clean"),
    ("checkpoints/model_adv_v1.pt", "adv_v1"),
    ("checkpoints/model_adv_v2.pt", "adv_v2"),
]


def test_one_seed(seed, df, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)

    # 生成对抗样本
    samples = []
    for _, row in df.iterrows():
        text = str(row['text'])
        adv_text = generate_adversarial(text, max_swaps=2)
        samples.append({
            'text': text,
            'event': int(row['event']),
            'true_label': int(row['label']),
            'adv_text': adv_text,
            'perturbed': adv_text != text,
        })

    results = []
    for path, label in MODELS:
        model, tokenizer = load_dl(path, device=device)
        _init_predictor(model, tokenizer, device=device)

        clean_correct = 0
        flipped, total_adv = 0, 0
        high_conf_flipped = 0

        for s in samples:
            clean_pred = predict(s['text'], s['event'])
            if clean_pred['label'] == s['true_label']:
                clean_correct += 1
            if not s['perturbed']:
                continue
            adv_pred = predict(s['adv_text'], s['event'])
            total_adv += 1
            if clean_pred['label'] != adv_pred['label']:
                flipped += 1
                if clean_pred['confidence'] > 0.9:
                    high_conf_flipped += 1

        results.append({
            'label': label,
            'clean_acc': clean_correct / len(samples) * 100,
            'flip_rate': flipped / total_adv * 100 if total_adv > 0 else 0,
            'flipped': flipped,
            'high_conf': high_conf_flipped,
        })

        del model

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=100)
    args = parser.parse_args()

    device = "cpu"
    df = pd.read_csv("rumer2026/val.csv")
    if args.n_samples < len(df):
        df = df.head(args.n_samples)
    print(f"样本: {len(df)} 条, 种子: {SEEDS}")
    print()

    # 收集所有结果
    all_runs = {s: None for s in SEEDS}
    for seed in SEEDS:
        print(f"seed={seed}...", end=" ", flush=True)
        t0 = time.time()
        all_runs[seed] = test_one_seed(seed, df, device)
        print(f"({time.time()-t0:.0f}s)")

    # 汇总
    print(f"\n{'='*80}")
    print("多种子汇总")
    print(f"{'='*80}")
    header = f"{'模型':<10} {'指标':<6}"
    for s in SEEDS:
        header += f" {'seed='+str(s):>10}"
    header += f" {'均值':>8} {'±std':>6}"
    print(header)
    print("-" * len(header))

    for i, (_, label) in enumerate(MODELS):
        for metric, key in [("Acc", "clean_acc"), ("翻转率", "flip_rate"), ("高置信", "high_conf")]:
            vals = [all_runs[s][i][key] for s in SEEDS]
            avg = np.mean(vals)
            std = np.std(vals)
            row = f"{label:<10} {metric:<6}"
            for v in vals:
                row += f" {v:>10.1f}" if key != "high_conf" else f" {v:>10}"
            row += f" {avg:>8.1f}" if key != "high_conf" else f" {avg:>8.0f}"
            row += f" ±{std:.1f}" if key != "high_conf" else f" ±{std:.0f}"
            print(row)
        print()

    # 推荐报告用数据
    print("=" * 40)
    print("报告建议: 用 seed=42 作为主数据, 标注均值±std 展现稳定性")


if __name__ == "__main__":
    main()
