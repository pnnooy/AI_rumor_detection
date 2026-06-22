"""分析 LLM 推理结果 (需先运行 inference.py 生成结果)"""
import pandas as pd


def main():
    df = pd.read_csv('results/val_results_full.csv')
    total = len(df)

    # LLM 调用统计
    failed = df['explanation'].str.startswith('[LLM', na=False).sum()
    success = total - failed

    print(f"总样本: {total}")
    print(f"LLM解释成功: {success} ({success/total*100:.1f}%)")
    print(f"LLM调用失败: {failed} ({failed/total*100:.1f}%)")

    ok = df[~df['explanation'].str.startswith('[LLM', na=False)]
    if len(ok) > 0:
        avg_len = ok['explanation'].str.len().mean()
        lens = ok['explanation'].str.len()
        print(f"成功解释平均长度: {avg_len:.0f} 字符")
        print(f"长度分布: min={lens.min()}, max={lens.max()}, median={lens.median():.0f}")
        short_count = (lens < 50).sum()
        print(f"过短解释 (<50字): {short_count} 条")

    # 分类准确率
    correct = (df['true_label'] == df['pred_label']).sum()
    print(f"\n分类准确率: {correct}/{total} = {correct/total*100:.2f}%")

    # 按事件统计失败率
    print(f"\n=== 各事件 LLM 拦截率 ===")
    for event_id in sorted(df['event'].unique()):
        subset = df[df['event'] == event_id]
        event_failed = subset['explanation'].str.startswith('[LLM', na=False).sum()
        print(f"Event {int(event_id)}: {event_failed}/{len(subset)} 失败 ({event_failed/len(subset)*100:.1f}%)")

    print(f"\n=== 成功样例 (前3条) ===")
    for i in range(min(3, len(ok))):
        row = ok.iloc[i]
        print(f"[{i+1}] text: {str(row['text'])[:80]}")
        print(f"    true={int(row['true_label'])} pred={int(row['pred_label'])} conf={row['confidence']:.4f}")
        print(f"    keywords: {row['keywords']}")
        print(f"    LLM: {str(row['explanation'])[:250]}")
        print()

    if failed > 0:
        bad = df[df['explanation'].str.startswith('[LLM', na=False)]
        print(f"=== 失败样例 (前5条) ===")
        for i in range(min(5, len(bad))):
            row = bad.iloc[i]
            print(f"[{i+1}] text: {str(row['text'])[:80]}")
            print(f"    error: {str(row['explanation'])[:200]}")
            print()

    # 检查完整解释文本
    print(f"=== 完整解释验证 (随机3条) ===")
    sample = ok.sample(min(3, len(ok)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows()):
        full = str(row['explanation'])
        print(f"[{i+1}] len={len(full)} | {full}")
        print()

    # 检查是否有截断
    print(f"=== 截断检查 (随机10条) ===")
    sample2 = ok.sample(min(10, len(ok)), random_state=7)
    truncated = 0
    for i, (_, row) in enumerate(sample2.iterrows()):
        full = str(row['explanation'])
        ends_ok = full.strip().endswith(('。', '.', '！', '?', '？'))
        if not ends_ok:
            truncated += 1
            print(f"[{i+1}] 疑似截断 len={len(full)}: ...{full[-30:]}")
    print(f"疑似截断: {truncated}/{len(sample2)} 条")


if __name__ == '__main__':
    main()
