"""分析 V3 结果：LLM 独立判断 vs DL 预测 vs 真实标签"""
import pandas as pd
import re

df = pd.read_csv('results/val_results_v3.csv')
total = len(df)

def parse_llm_verdict(text):
    """从 LLM 解释文本中提取独立判断: 1=不实信息, 0=真实信息, -1=无法解析"""
    if not isinstance(text, str) or text.startswith('[LLM'):
        return -1
    # 找最终结论区域
    # 新模式: "这条推文更可能是"不实信息"/"真实信息""
    # 旧模式: "判定为谣言/非谣言"
    patterns_rumor = [
        r'更可能是.{0,4}不实信息',  # 新prompt
        r'最终结论.{0,10}不实信息',
        r'判定为\s*谣言',
        r'判定为\s*不实信息',
        r'更可能是\s*谣言',
        r'属于\s*谣言',
        r'归为\s*谣言',
        r'最终判定.{0,5}谣言',
    ]
    patterns_real = [
        r'更可能是.{0,4}真实信息',
        r'最终结论.{0,10}真实信息',
        r'判定为\s*非谣言',
        r'判定为\s*真实信息',
        r'更可能是\s*非谣言',
        r'属于\s*真实',
        r'归为\s*真实',
        r'最终判定.{0,5}非谣言',
        r'最终判定.{0,5}真实',
    ]

    rumor_score = sum(1 for p in patterns_rumor if re.search(p, text))
    real_score = sum(1 for p in patterns_real if re.search(p, text))

    if rumor_score > real_score:
        return 1
    elif real_score > rumor_score:
        return 0
    else:
        # 无法明确判断，尝试找 "我同/不同意"
        if '不同意' in text or '不认同' in text:
            # 说不同意但没说改判什么，看上下文
            if '真实' in text[-100:] and '不实' not in text[-100:]:
                return 0
            if '不实' in text[-100:] or '谣言' in text[-100:]:
                return 1
        return -1

# 解析
df['llm_verdict'] = df['explanation'].apply(parse_llm_verdict)

# 统计
parsed = df[df['llm_verdict'] >= 0]
unparsed = len(df) - len(parsed)
print(f"总样本: {total}")
print(f"LLM判断可解析: {len(parsed)} ({len(parsed)/total*100:.1f}%)")
print(f"无法解析: {unparsed}")

# DL vs LLM 对比
agreement = (parsed['pred_label'] == parsed['llm_verdict']).sum()
disagreement = len(parsed) - agreement
print(f"\nDL vs LLM:")
print(f"  一致: {agreement} ({agreement/len(parsed)*100:.1f}%)")
print(f"  不一致: {disagreement} ({disagreement/len(parsed)*100:.1f}%)")

# 准确率对比
dl_correct = (parsed['pred_label'] == parsed['true_label']).sum()
llm_correct = (parsed['llm_verdict'] == parsed['true_label']).sum()
print(f"\n准确率:")
print(f"  DL模型: {dl_correct}/{len(parsed)} = {dl_correct/len(parsed)*100:.2f}%")
print(f"  LLM判断: {llm_correct}/{len(parsed)} = {llm_correct/len(parsed)*100:.2f}%")
print(f"  差异: {llm_correct - dl_correct:+d} 条 ({(llm_correct-dl_correct)/len(parsed)*100:+.2f}%)")

# DL错但LLM改对的
dl_wrong = parsed[parsed['pred_label'] != parsed['true_label']]
llm_fixed = dl_wrong[dl_wrong['llm_verdict'] == dl_wrong['true_label']]
print(f"\nDL错误 LLM纠正: {len(llm_fixed)}/{len(dl_wrong)} 条")

# DL对但LLM改错的
dl_right = parsed[parsed['pred_label'] == parsed['true_label']]
llm_broke = dl_right[dl_right['llm_verdict'] != dl_right['true_label']]
print(f"DL正确 LLM改错: {len(llm_broke)}/{len(dl_right)} 条")

# 展示LLM改判案例
print(f"\n=== LLM 改判案例 (前10条) ===")
changed = parsed[parsed['pred_label'] != parsed['llm_verdict']]
for i, (_, row) in enumerate(changed.head(10).iterrows()):
    dl_label = "不实" if row['pred_label']==1 else "真实"
    llm_label = "不实" if row['llm_verdict']==1 else "真实"
    true_label = "不实" if row['true_label']==1 else "真实"
    result = "[OK]改对" if row['llm_verdict']==row['true_label'] else "[FAIL]改错"
    print(f"[{i+1}] {result} | DL={dl_label} -> LLM={llm_label} | true={true_label} | conf={row['confidence']:.3f}")
    print(f"    text: {str(row['text'])[:80]}")
    print(f"    LLM: {str(row['explanation'])[:150]}")
    print()
