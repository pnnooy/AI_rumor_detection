"""
评估脚本 — 韩宇飞 实现

用法:
    python evaluate.py --input results/val_results.csv
    python evaluate.py --input results/val_results.csv --output-dir figures/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端，服务器也可用
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


def compute_metrics(df: pd.DataFrame):
    """计算整体和按事件分组的所有指标"""
    y_true = df['true_label'].astype(int)
    y_pred = df['pred_label'].astype(int)

    metrics = {}

    # 整体指标
    metrics['overall'] = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }

    # 按事件分组
    if 'event' in df.columns:
        for event_id in sorted(df['event'].unique()):
            mask = df['event'] == event_id
            if mask.sum() == 0:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            metrics[f'event_{event_id}'] = {
                'accuracy':  accuracy_score(yt, yp),
                'precision': precision_score(yt, yp, zero_division=0),
                'recall':    recall_score(yt, yp, zero_division=0),
                'f1':        f1_score(yt, yp, zero_division=0),
                'n_samples': mask.sum(),
            }

    return metrics


def print_metrics(metrics: dict, df: pd.DataFrame):
    """打印评估指标表格"""
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    # 整体
    m = metrics['overall']
    print(f"\n  整体:")
    print(f"    Accuracy:  {m['accuracy']:.4f}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1:        {m['f1']:.4f}")

    # 各事件
    event_keys = [k for k in metrics if k.startswith('event_')]
    if event_keys:
        print(f"\n  {'Event':<10} {'样本':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print(f"  {'-'*48}")
        for key in event_keys:
            m = metrics[key]
            event_id = key.replace('event_', '')
            print(f"  Event {event_id:<5} {m['n_samples']:>6} {m['accuracy']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    print(f"\n  分类报告:\n")
    print(classification_report(
        df['true_label'].astype(int),
        df['pred_label'].astype(int),
        target_names=['Non-rumor', 'Rumor'],
        digits=4
    ))


def plot_confusion_matrix(df: pd.DataFrame, output_dir: str):
    """混淆矩阵热力图"""
    cm = confusion_matrix(df['true_label'], df['pred_label'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-rumor', 'Rumor'],
                yticklabels=['Non-rumor', 'Rumor'],
                annot_kws={'size': 20})
    plt.xlabel('预测', fontsize=13)
    plt.ylabel('真实', fontsize=13)
    plt.title('混淆矩阵', fontsize=15)
    plt.tight_layout()
    path = f'{output_dir}/confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 混淆矩阵 -> {path}")


def plot_event_accuracy(metrics: dict, output_dir: str):
    """各事件准确率柱状图"""
    event_keys = [k for k in metrics if k.startswith('event_')]
    if not event_keys:
        return

    events = [k.replace('event_', '') for k in event_keys]
    accs = [metrics[k]['accuracy'] for k in event_keys]
    f1s = [metrics[k]['f1'] for k in event_keys]
    overall_acc = metrics['overall']['accuracy']

    x = np.arange(len(events))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1', color='#DD8452')

    ax.axhline(y=overall_acc, color='#4C72B0', linestyle='--', alpha=0.5,
               label=f'整体 Acc: {overall_acc:.3f}')
    ax.set_xlabel('事件', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各事件准确率与 F1', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Event {e}' for e in events])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    # 数值标注
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = f'{output_dir}/event_accuracy.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 事件准确率 -> {path}")


def plot_confidence_histogram(df: pd.DataFrame, output_dir: str):
    """置信度分布直方图（正确 vs 错误）"""
    df = df.copy()
    df['correct'] = (df['true_label'] == df['pred_label']).map(
        {True: '正确', False: '错误'})

    plt.figure(figsize=(8, 5))
    for label, color in [('正确', '#4C72B0'), ('错误', '#DD8452')]:
        subset = df[df['correct'] == label]['confidence']
        plt.hist(subset, bins=20, alpha=0.6, label=label, color=color)

    plt.xlabel('置信度', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.title('置信度分布：正确预测 vs 错误预测', fontsize=14)
    plt.legend()
    plt.tight_layout()
    path = f'{output_dir}/confidence_hist.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 置信度分布 -> {path}")


def plot_calibration_curve(df: pd.DataFrame, output_dir: str):
    """置信度校准曲线（Reliability Diagram）"""
    # 仅分析预测为Rumor的样本
    rumor_df = df[df['pred_label'] == 1].copy()
    if len(rumor_df) < 10:
        print("  [WARN] 预测为Rumor的样本太少，跳过校准曲线")
        return

    rumor_df['conf_bin'] = pd.cut(rumor_df['confidence'],
                                   bins=np.arange(0, 1.05, 0.1),
                                   labels=np.arange(0.05, 1.0, 0.1))
    calibration = rumor_df.groupby('conf_bin', observed=False).agg(
        accuracy=('true_label', 'mean'),
        count=('true_label', 'count')
    ).dropna()

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='完美校准')
    plt.scatter(
        calibration.index.astype(float), calibration['accuracy'],
        s=calibration['count'] * 10, alpha=0.6, color='#4C72B0'
    )
    plt.xlabel('置信度', fontsize=12)
    plt.ylabel('实际Rumor比例', fontsize=12)
    plt.title('置信度校准曲线（气泡大小 = 样本量）', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    path = f'{output_dir}/calibration_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 校准曲线 -> {path}")


def plot_event_f1_comparison(metrics: dict, output_dir: str):
    """各事件 Precision / Recall / F1 对比"""
    event_keys = [k for k in metrics if k.startswith('event_')]
    if not event_keys:
        return

    events = [k.replace('event_', '') for k in event_keys]
    precs = [metrics[k]['precision'] for k in event_keys]
    recalls = [metrics[k]['recall'] for k in event_keys]
    f1s = [metrics[k]['f1'] for k in event_keys]

    x = np.arange(len(events))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precs, width, label='Precision', color='#4C72B0')
    ax.bar(x, recalls, width, label='Recall', color='#55A868')
    ax.bar(x + width, f1s, width, label='F1', color='#DD8452')

    ax.set_xlabel('事件', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各事件 Precision / Recall / F1 对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Event {e}' for e in events])
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = f'{output_dir}/event_f1_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] F1 对比 -> {path}")


def analyze_confidence_tiers(df: pd.DataFrame):
    """
    按置信度分级统计准确率，并打印简要结果
    """
    if 'confidence' not in df.columns:
        print("  [WARN] 结果 CSV 中无 confidence 列，跳过分级统计")
        return

    df = df.copy()
    df['tier'] = df['confidence'].apply(
        lambda c: '确信(≥0.9)' if c >= 0.9 else ('倾向(0.7-0.9)' if c >= 0.7 else '存疑(<0.7)')
    )

    print(f"\n  === 置信度分级统计 ===")
    print(f"  {'分级':<18} {'样本':>8} {'准确率':>10}")
    print(f"  {'-' * 45}")
    for tier in ['确信(≥0.9)', '倾向(0.7-0.9)', '存疑(<0.7)']:
        subset = df[df['tier'] == tier]
        if len(subset) > 0:
            acc = (subset['true_label'] == subset['pred_label']).mean()
            pct = len(subset) / len(df) * 100
            print(f"  {tier:<18} {len(subset):>4} ({pct:>5.1f}%)  {acc:>8.1%}")
        else:
            print(f"  {tier:<18}    0 (  0.0%)       -")
    print("  ===")


def plot_confidence_tier_accuracy(df: pd.DataFrame, output_dir: str):
    """置信度三级柱状图 + 饼图（并排显示）"""
    if 'confidence' not in df.columns:
        return

    df = df.copy()
    df['tier'] = df['confidence'].apply(
        lambda c: '确信(≥0.9)' if c >= 0.9 else ('倾向(0.7-0.9)' if c >= 0.7 else '存疑(<0.7)')
    )
    order = ['确信(≥0.9)', '倾向(0.7-0.9)', '存疑(<0.7)']
    colors = ['#55A868', '#DD8452', '#C44E52']

    counts = []
    accs = []
    for tier in order:
        subset = df[df['tier'] == tier]
        counts.append(len(subset))
        if len(subset) > 0:
            accs.append((subset['true_label'] == subset['pred_label']).mean() * 100)
        else:
            accs.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左侧：准确率柱状图
    bars = ax1.bar(order, accs, color=colors, alpha=0.85)
    for bar, cnt, acc in zip(bars, counts, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{acc:.1f}% (n={cnt})",
            ha='center', va='bottom', fontsize=10
        )
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('各置信度分级准确率', fontsize=13)
    ax1.axhline(y=((df['true_label'] == df['pred_label']).mean() * 100),
                color='gray', linestyle='--', alpha=0.6, label='整体准确率')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.2)

    # 右侧：样本占比饼图
    sizes = [c for c in counts if c > 0]
    labels = [o for o, c in zip(order, counts) if c > 0]
    palette = [colors[i] for i, c in enumerate(counts) if c > 0]
    if sizes:
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                colors=palette, startangle=90, textprops={'fontsize': 11})
        ax2.set_title('各置信度分级样本占比', fontsize=13)

    plt.tight_layout()
    path = f'{output_dir}/confidence_tiers.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 置信度分级图 -> {path}")


def evaluate(results_csv: str, output_dir: str = "figures"):
    """
    完整评估管道

    Args:
        results_csv: inference.py 输出的结果 CSV
        output_dir:  图表保存目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 加载
    print(f"[1/7] 加载结果: {results_csv}")
    df = pd.read_csv(results_csv)
    required_cols = ['true_label', 'pred_label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"结果 CSV 缺少必需列: {col}")
    print(f"  共 {len(df)} 条记录")

    # 计算指标
    print("[2/7] 计算指标...")
    metrics = compute_metrics(df)
    print_metrics(metrics, df)

    # 绘图
    print("[3/7] 绘制混淆矩阵...")
    plot_confusion_matrix(df, output_dir)

    print("[4/7] 绘制事件准确率...")
    plot_event_accuracy(metrics, output_dir)

    print("[5/7] 绘制置信度分析...")
    if 'confidence' in df.columns:
        plot_confidence_histogram(df, output_dir)
        plot_calibration_curve(df, output_dir)
        plot_confidence_tier_accuracy(df, output_dir)
    else:
        print("  [WARN] 结果中无 confidence 列，跳过置信度分析")

    print("[6/7] 绘制 F1 对比...")
    plot_event_f1_comparison(metrics, output_dir)

    # 置信度分级统计
    print("[7/7] 置信度分级统计...")
    analyze_confidence_tiers(df)

    # 错误分析
    analyze_errors(df)

    print(f"\n{'='*70}")
    print(f"评估完成，图表保存在 {output_dir}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Rumor检测系统 — 评估脚本"
    )
    parser.add_argument(
        '--input', required=True,
        help='inference.py 输出的结果 CSV'
    )
    parser.add_argument(
        '--output-dir', default='figures',
        help='图表保存目录 (默认: figures/)'
    )
    args = parser.parse_args()
    evaluate(args.input, args.output_dir)


if __name__ == '__main__':
    main()
