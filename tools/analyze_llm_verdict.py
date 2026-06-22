"""分析 LLM 独立判断 vs DL 预测 vs 真实标签 (需先生成 V3 结果)"""
import pandas as pd
import re


def parse_llm_verdict(text):
    """从 LLM 解释文本中提取独立判断: 1=不实信息, 0=真实信息, -1=无法解析"""
    if not isinstance(text, str) or text.startswith('[LLM'):
        return -1
    patterns_rumor = [
        r'更可能是.{0,4}不实信息',
        r'最终结论.{0,10}不实信息',
        r'判定为\s*谣言', r'判定为\s*不实信息',
        r'更可能是\s*谣言', r'属于\s*谣言', r'归为\s*谣言',
        r'最终判定.{0,5}谣言',
    ]
    patterns_real = [
        r'更可能是.{0,4}真实信息',
        r'最终结论.{0,10}真实信息',
        r'判定为\s*非谣言', r'判定为\s*真实信息',
        r'更可能是\s*非谣言', r'属于\s*真实', r'归为\s*真实',
        r'最终判定.{0,5}非谣言', r'最终判定.{0,5}真实',
    ]
    rumor_score = sum(1 for p in patterns_rumor if re.search(p, text))
    real_score = sum(1 for p in patterns_real if re.search(p, text))
    if rumor_score > real_score:
        return 1
    elif real_score > rumor_score:
        return 0
    else:
        if '不同意' in text or '不认同' in text:
            if '真实' in text[-100:] and '不实' not in text[-100:]:
                return 0
            if '不实' in text[-100:] or '谣言' in text[-100:]:
                return 1
        return -1


def main():
    df = pd.read_csv('results/val_results_v3.csv')
    total = len(df)
    df['llm_verdict'] = df['explanation'].apply(parse_llm_verdict)

    parsed = df[df['llm_verdict'] >= 0]
    unparsed = len(df) - len(parsed)
    print(f"总样本: {total}")
    print(f"LLM判断可解析: {len(parsed)} ({len(parsed)/total*100:.1f}%)")
    print(f"无法解析: {unparsed}")

    agreement = (parsed['pred_label'] == parsed['llm_verdict']).sum()
    print(f"\nDL vs LLM: 一致 {agreement} ({agreement/len(parsed)*100:.1f}%)")

    dl_correct = (parsed['pred_label'] == parsed['true_label']).sum()
    llm_correct = (parsed['llm_verdict'] == parsed['true_label']).sum()
    print(f"\n准确率: DL={dl_correct/len(parsed)*100:.2f}%  LLM={llm_correct/len(parsed)*100:.2f}%")

    dl_wrong = parsed[parsed['pred_label'] != parsed['true_label']]
    llm_fixed = dl_wrong[dl_wrong['llm_verdict'] == dl_wrong['true_label']]
    print(f"\nDL错误 LLM纠正: {len(llm_fixed)}/{len(dl_wrong)} 条")

    dl_right = parsed[parsed['pred_label'] == parsed['true_label']]
    llm_broke = dl_right[dl_right['llm_verdict'] != dl_right['true_label']]
    print(f"DL正确 LLM改错: {len(llm_broke)}/{len(dl_right)} 条")

    print(f"\n=== LLM 改判案例 (前10条) ===")
    changed = parsed[parsed['pred_label'] != parsed['llm_verdict']]
    for i, (_, row) in enumerate(changed.head(10).iterrows()):
        dl_label = "不实" if row['pred_label'] == 1 else "真实"
        llm_label = "不实" if row['llm_verdict'] == 1 else "真实"
        true_label = "不实" if row['true_label'] == 1 else "真实"
        result = "[OK]改对" if row['llm_verdict'] == row['true_label'] else "[FAIL]改错"
        print(f"[{i+1}] {result} | DL={dl_label} -> LLM={llm_label} | true={true_label}")
        print(f"    text: {str(row['text'])[:80]}")
        print(f"    LLM: {str(row['explanation'])[:150]}")
        print()


if __name__ == '__main__':
    main()
