"""
对抗样本鲁棒性分析 — 韩宇飞 实现

对 DL 分类器进行词级同义词替换攻击，评估系统鲁棒性。

用法:
    python adversarial.py --input results/val_results.csv --original rumer2026/val.csv
"""

import argparse
import random
import pandas as pd
from collections import Counter

# ============================================================
# 对抗样本生成
# ============================================================

def get_synonym(word: str) -> str | None:
    """
    获取 word 的一个同义词（基于 WordNet）

    Args:
        word: 英文单词

    Returns:
        同义词，无则返回 None
    """
    try:
        from nltk.corpus import wordnet
    except ImportError:
        print("请先安装 nltk 和下载 WordNet:")
        print("  pip install nltk")
        print("  python -c 'import nltk; nltk.download(\"wordnet\"); nltk.download(\"averaged_perceptron_tagger\")'")
        return None

    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    lemmas = synsets[0].lemma_names()
    synonyms = [l.replace('_', ' ') for l in lemmas
                if l.lower() != word.lower()]
    return random.choice(synonyms) if synonyms else None


def generate_adversarial(text: str, max_swaps: int = 2, seed: int = 42) -> str:
    """
    对一条推文生成对抗样本（替换 1-2 个词为同义词）

    Args:
        text:      原推文文本
        max_swaps: 最多替换几个词
        seed:      随机种子（保证可复现）

    Returns:
        对抗样本文本
    """
    rng = random.Random(seed)
    words = text.split()

    # 收集有同义词的词位置
    candidates = []
    for i, w in enumerate(words):
        # 跳过短词、非纯字母词、全大写缩写
        if len(w) <= 3 or not w.isalpha() or w.isupper():
            continue
        syn = get_synonym(w)
        if syn:
            candidates.append(i)

    if not candidates:
        return text  # 无法生成对抗样本

    # 最多替换 max_swaps 个词
    n_swaps = min(max_swaps, len(candidates))
    swap_indices = rng.sample(candidates, n_swaps)

    new_words = words.copy()
    for idx in swap_indices:
        syn = get_synonym(new_words[idx])
        if syn:
            new_words[idx] = syn

    return ' '.join(new_words)


# ============================================================
# 鲁棒性分析
# ============================================================

def analyze_robustness(results_csv: str, original_csv: str, output_dir: str = "results"):
    """
    对抗样本鲁棒性分析主函数

    Args:
        results_csv:  inference.py 在原始验证集上的输出
        original_csv: 原始验证集 CSV (rumer2026/val.csv)
        output_dir:   输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始结果和数据
    results = pd.read_csv(results_csv)
    original = pd.read_csv(original_csv)

    print("=" * 60)
    print("对抗样本鲁棒性分析")
    print("=" * 60)

    # 1. 生成对抗样本
    print("\n[1/4] 生成对抗样本...")
    adversarial_texts = []
    n_skipped = 0
    for text in original['text']:
        adv = generate_adversarial(str(text))
        if adv == str(text):
            n_skipped += 1
        adversarial_texts.append(adv)
    print(f"  生成 {len(adversarial_texts)} 条对抗样本 ({n_skipped} 条无法扰动)")

    # 保存对抗样本
    adv_df = original.copy()
    adv_df['text'] = adversarial_texts
    adv_df.to_csv(f'{output_dir}/adversarial_samples.csv', index=False)
    print(f"  对抗样本已保存至 {output_dir}/adversarial_samples.csv")

    # 2. 提示：需要用 inference.py 对对抗样本重新推理
    print(f"\n[2/4] 下一步: 用 inference.py 对对抗样本推理")
    print(f"  python inference.py --input {output_dir}/adversarial_samples.csv "
          f"--output {output_dir}/adversarial_results.csv --no-llm")

    # 3. 提示：如果有对抗样本的结果，做对比分析
    print(f"\n[3/4] 对比分析（需先完成第 2 步）")
    adv_results_path = f'{output_dir}/adversarial_results.csv'
    if os.path.exists(adv_results_path):
        adv_results = pd.read_csv(adv_results_path)
        compare_robustness(results, adv_results, output_dir)
    else:
        print(f"  等待 {adv_results_path} 生成后运行对比分析")

    # 4. 脆弱模式分析
    print(f"\n[4/4] 脆弱模式分析")
    analyze_vulnerability_patterns(results, adversarial_texts, output_dir)


def compare_robustness(original_results: pd.DataFrame,
                       adversarial_results: pd.DataFrame,
                       output_dir: str):
    """对比原始预测和对抗样本预测"""
    import matplotlib.pyplot as plt

    # 对齐行数
    n = min(len(original_results), len(adversarial_results))
    orig = original_results.head(n)
    adv = adversarial_results.head(n)

    # 统计翻转
    flipped = (orig['pred_label'] != adv['pred_label'])
    flip_rate = flipped.mean()
    print(f"\n  攻击成功率（label 翻转率）: {flip_rate:.1%}")

    # 翻转方向
    fp_flips = ((orig['pred_label'] == 0) & (adv['pred_label'] == 1)).sum()
    fn_flips = ((orig['pred_label'] == 1) & (adv['pred_label'] == 0)).sum()
    print(f"  误报翻转 (0→1): {fp_flips}")
    print(f"  漏报翻转 (1→0): {fn_flips}")

    # 按事件分组的翻转率
    if 'event' in orig.columns:
        print(f"\n  各事件攻击成功率:")
        for e in sorted(orig['event'].unique()):
            mask = orig['event'] == e
            e_flip = flipped[mask].mean()
            n_e = mask.sum()
            print(f"    Event {int(e)}: {e_flip:.1%} ({n_e} 条)")


def analyze_vulnerability_patterns(results: pd.DataFrame,
                                    adversarial_texts: list,
                                    output_dir: str):
    """分析哪些样本更容易被攻击"""
    # 统计被扰动替换最多的词
    # TODO: 在完整实现中对比原文本和对抗文本的词差异

    # 按原置信度分组
    if 'confidence' in results.columns:
        print(f"\n  原置信度分布（可能影响攻击成功率）:")
        results['conf_bin'] = pd.cut(
            results['confidence'],
            bins=[0, 0.5, 0.7, 0.9, 1.0],
            labels=['<0.5', '0.5-0.7', '0.7-0.9', '>0.9']
        )
        print(results['conf_bin'].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(
        description="对抗样本鲁棒性分析"
    )
    parser.add_argument('--input', required=True,
                        help='inference.py 在原始验证集上的输出 CSV')
    parser.add_argument('--original', default='rumer2026/val.csv',
                        help='原始验证集 CSV')
    parser.add_argument('--output-dir', default='results',
                        help='输出目录')
    args = parser.parse_args()

    analyze_robustness(args.input, args.original, args.output_dir)


if __name__ == '__main__':
    main()
