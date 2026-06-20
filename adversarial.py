"""
对抗样本攻击与防护 — 姜新晨 实现

六阶段完整攻防评估管道:
  1. 攻击生成：WordNet 同义词替换对抗样本
  2. 自动推理：对对抗样本做 DL 分类
  3. 对比分析：原始 vs 对抗预测对比
  4. LLM 交叉验证：对翻转样本独立判断（防护亮点）
  5. 四组实验：无防护 / 对抗训练 / LLM 交叉验证 / 双重防护
  6. CLI 入口：一键运行完整评估

用法:
    # 完整攻防评估（一条命令跑完）
    python adversarial.py --mode full --original rumer2026/val.csv --use-llm-defense

    # 仅生成对抗样本
    python adversarial.py --mode generate --original rumer2026/val.csv

    # 仅评估（需要已有对抗样本推理结果）
    python adversarial.py --mode evaluate \\
        --input results/val_results.csv \\
        --adversarial results/adversarial_results.csv
"""

import os
import sys
import argparse
import random
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# 对抗样本生成（WordNet 同义词替换）
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
        print("[WARN] 请先安装 nltk 和下载 WordNet:")
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
# 自动推理（Phase 3.1）
# ============================================================

def run_inference_on_adversarial(
    adversarial_csv: str,
    output_csv: str,
    model_path: str = "checkpoints/best_model.pt",
    device: str = "cpu"
) -> pd.DataFrame:
    """
    用现有分类器对对抗样本做推理（仅 DL 分类，不调 LLM）

    步骤:
    1. 加载模型 load_model(model_path, device)
    2. 逐条调用 predict(text, event_id)
    3. 保存结果 CSV
    4. 返回 DataFrame

    关键注意事项:
    - 必须按原始 event_id 传入 predict()，否则 [EVENT_N] token 会错配
    - 使用 --no-llm 等效逻辑（纯分类，不调 API）
    - 进度条用 tqdm 或每 50 条打印一次
    """
    from models.classifier import load_model
    from models.keyword_extractor import predict

    print(f"\n{'='*60}")
    print("[推理] 对对抗样本做 DL 分类")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  输入: {adversarial_csv}")
    print(f"  输出: {output_csv}")

    # 加载模型
    model, tokenizer = load_model(model_path, device=device)

    # 加载对抗样本
    df = pd.read_csv(adversarial_csv)
    print(f"  样本数: {len(df)}")

    # 逐条推理
    results = []
    n_total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row['text'])
        event_id = int(row['event'])
        dl_result = predict(text, event_id)

        results.append({
            'id': row.get('id', i),
            'text': text,
            'event': event_id,
            'true_label': int(row['label']),
            'pred_label': dl_result['label'],
            'confidence': dl_result['confidence'],
            'keywords': ','.join([w for w, _ in dl_result['keywords']]),
            'explanation': '',  # 对抗样本不需要 LLM 解释
        })

        if (i + 1) % 50 == 0:
            print(f"  推理进度: {i+1}/{n_total}")

    print(f"  推理进度: {n_total}/{n_total}")

    # 保存结果
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"  [OK] 推理结果已保存至 {output_csv}")

    return results_df


# ============================================================
# LLM 交叉验证防护（Phase 4）
# ============================================================

CROSS_VAL_PROMPT = """你是一个独立的谣言事实核查员。请仅根据以下信息判断这条推文是否在传播谣言。

[事件背景]
{event_context}

[推文内容]
"{text}"

[已知信息]
- 一个深度学习模型将这条推文判定为{model_prediction}（置信度 {confidence:.0%}）
- 但你可能面对的是被恶意修改过的推文文本（通过同义词替换等方式绕过检测）
- 请基于你的常识和对该事件背景的理解，独立判断这条推文是否在传播虚假信息

请用以下格式回复：
判断: [谣言/非谣言/不确定]
理由: [一句话说明判断依据]
"""


def llm_cross_validation(
    text: str,
    dl_pred_label: int,
    dl_confidence: float,
    original_pred_label: int,
    event_context: str = "",
    explainer: "LLMExplainer | None" = None
) -> dict:
    """
    LLM 交叉验证：当 DL 预测不一致时，让 LLM 独立判断

    调用流程:
    1. 构造专门的法律/事实核查 prompt（不同于解释 prompt）
    2. LLM 基于事件背景 + 常识判断推文真实性
    3. 返回 {'llm_label': int, 'llm_reasoning': str, 'verdict': str}

    verdict 取值:
    - "支持DL"  — LLM 同意 DL 判断
    - "推翻DL"  — LLM 认为 DL 判断错误
    - "不确定"  — LLM 无法确定，建议人工复核

    关键设计:
    - prompt 必须强调 LLM 是独立判断，不受 DL 结果影响
    - temperature=0.1（比解释更确定）
    - 如果 LLM 调用失败 → verdict="不确定"（保守策略，不强行纠错）
    """
    if explainer is None:
        return {
            'llm_label': -1,
            'llm_reasoning': 'LLM explainer 未初始化',
            'verdict': '不确定'
        }

    try:
        # 构造 prompt
        model_prediction = "谣言" if dl_pred_label == 1 else "非谣言"
        prompt = CROSS_VAL_PROMPT.format(
            event_context=event_context or "暂无该事件的背景信息。",
            text=text,
            model_prediction=model_prediction,
            confidence=dl_confidence
        )

        # 调用 LLM
        response = explainer.client.chat.completions.create(
            model=explainer.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的谣言事实核查员。请独立判断推文真实性，不受他人结论影响。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=256
        )

        raw_output = response.choices[0].message.content.strip()

        # 解析 LLM 输出
        llm_label = -1  # 默认不确定
        llm_reasoning = raw_output

        # 解析"判断:"行
        for line in raw_output.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('判断:') or line_stripped.startswith('判断：'):
                judgment = line_stripped.split(':', 1)[-1].split('：', 1)[-1].strip()
                if '谣言' in judgment and '非谣言' not in judgment:
                    llm_label = 1
                elif '非谣言' in judgment or '非谣言' in judgment:
                    llm_label = 0
                elif '不确定' in judgment:
                    llm_label = -1
                break

        # 确定 verdict
        if llm_label == -1:
            verdict = '不确定'
        elif llm_label == dl_pred_label:
            verdict = '支持DL'
        else:
            verdict = '推翻DL'

        return {
            'llm_label': llm_label,
            'llm_reasoning': llm_reasoning,
            'verdict': verdict
        }

    except Exception as e:
        print(f"  [WARN] LLM 交叉验证失败: {e}")
        return {
            'llm_label': -1,
            'llm_reasoning': f'LLM 调用失败: {str(e)[:100]}',
            'verdict': '不确定'
        }


# ============================================================
# 全自动鲁棒性分析管道（Phase 3.2）
# ============================================================

def analyze_robustness(
    results_csv: str = None,
    original_csv: str = "rumer2026/val.csv",
    output_dir: str = "results",
    model_path: str = "checkpoints/best_model.pt",
    device: str = "cpu",
    auto_run_inference: bool = True,
    use_llm_cross_validation: bool = False,
    adv_model_path: str = None,
):
    """
    全自动鲁棒性分析管道:

    Step 1: 生成对抗样本 → {output_dir}/adversarial/adversarial_samples.csv
    Step 2: 自动推理对抗样本 → {output_dir}/adversarial/adversarial_results.csv
    Step 3: 对比原始 vs 对抗预测，计算翻转率
    Step 4: LLM 交叉验证（可选）
    Step 5: 按事件/置信度分组分析脆弱性
    Step 6: 保存分析报告 → {output_dir}/adversarial/adversarial_report.txt

    所有步骤自动串联，无需人工干预。
    """
    adv_output_dir = os.path.join(output_dir, 'adversarial')
    os.makedirs(adv_output_dir, exist_ok=True)

    print("=" * 60)
    print("对抗样本鲁棒性分析 — 全自动管道")
    print("=" * 60)
    print(f"  原始数据: {original_csv}")
    print(f"  输出目录: {adv_output_dir}")
    print(f"  设备: {device}")
    print(f"  LLM 交叉验证: {'开启' if use_llm_cross_validation else '关闭'}")

    # ================================================================
    # Step 1: 生成对抗样本
    # ================================================================
    print(f"\n{'='*60}")
    print("[Step 1/5] 生成对抗样本...")
    print(f"{'='*60}")

    original = pd.read_csv(original_csv)
    adversarial_texts = []
    n_skipped = 0
    replaced_words_counter = Counter()

    for idx, row in original.iterrows():
        orig_text = str(row['text'])
        adv_text = generate_adversarial(orig_text)

        if adv_text == orig_text:
            n_skipped += 1
        else:
            # 统计被替换的词
            orig_words = set(orig_text.lower().split())
            adv_words = set(adv_text.lower().split())
            replaced = orig_words - adv_words
            for w in replaced:
                replaced_words_counter[w] += 1

        adversarial_texts.append(adv_text)

    print(f"  生成 {len(adversarial_texts)} 条对抗样本")
    print(f"  可扰动: {len(adversarial_texts) - n_skipped} 条")
    print(f"  无法扰动: {n_skipped} 条 ({n_skipped/len(adversarial_texts)*100:.1f}%)")

    # 保存对抗样本
    adv_df = original.copy()
    adv_df['text'] = adversarial_texts
    adv_sample_path = os.path.join(adv_output_dir, 'adversarial_samples.csv')
    adv_df.to_csv(adv_sample_path, index=False)
    print(f"  [OK] 对抗样本已保存至 {adv_sample_path}")

    # ================================================================
    # Step 2: 对对抗样本推理
    # ================================================================
    adv_results_path = os.path.join(adv_output_dir, 'adversarial_results.csv')

    if auto_run_inference:
        print(f"\n{'='*60}")
        print("[Step 2/5] 对对抗样本做 DL 推理...")
        print(f"{'='*60}")

        use_model = adv_model_path if adv_model_path else model_path
        adv_results = run_inference_on_adversarial(
            adversarial_csv=adv_sample_path,
            output_csv=adv_results_path,
            model_path=use_model,
            device=device
        )
    else:
        if os.path.exists(adv_results_path):
            print(f"\n[Step 2/5] 使用已有对抗样本推理结果: {adv_results_path}")
            adv_results = pd.read_csv(adv_results_path)
        else:
            print(f"\n[Step 2/5] 跳过（auto_run_inference=False 且文件不存在）")
            print(f"  提示: 手动运行:")
            print(f"  python inference.py --input {adv_sample_path} --output {adv_results_path} --no-llm")
            adv_results = None

    # ================================================================
    # Step 3: 对比分析
    # ================================================================
    print(f"\n{'='*60}")
    print("[Step 3/5] 对比原始预测 vs 对抗预测")
    print(f"{'='*60}")

    # 对原始验证集做推理（如果提供了 results_csv 就用它，否则自动推理）
    if results_csv and os.path.exists(results_csv):
        print(f"  使用已有原始推理结果: {results_csv}")
        orig_results = pd.read_csv(results_csv)
    else:
        print(f"  对原始验证集做推理...")
        orig_results = run_inference_on_adversarial(
            adversarial_csv=original_csv,
            output_csv=os.path.join(adv_output_dir, 'original_results.csv'),
            model_path=model_path,
            device=device
        )

    if adv_results is not None:
        comparison = compare_robustness(orig_results, adv_results, adv_output_dir)
    else:
        comparison = None

    # ================================================================
    # Step 4: LLM 交叉验证（仅对翻转样本）
    # ================================================================
    llm_verdicts = None
    if use_llm_cross_validation and adv_results is not None:
        print(f"\n{'='*60}")
        print("[Step 4/5] LLM 交叉验证防护")
        print(f"{'='*60}")

        try:
            from llm_explainer import LLMExplainer
            from event_context import get_event_context

            explainer = LLMExplainer()
            print(f"  [OK] LLMExplainer 初始化成功 (model={explainer.model})")

            # 找出翻转样本
            n = min(len(orig_results), len(adv_results))
            orig_preds = orig_results['pred_label'].values[:n]
            adv_preds = adv_results['pred_label'].values[:n]
            adv_confs = adv_results['confidence'].values[:n]
            orig_texts = original['text'].values[:n]
            events = original['event'].values[:n] if 'event' in original.columns else [0] * n

            flipped_mask = (orig_preds != adv_preds)
            flipped_indices = np.where(flipped_mask)[0]
            n_flipped = len(flipped_indices)

            print(f"  翻转样本数: {n_flipped}")
            print(f"  开始 LLM 交叉验证...")

            llm_verdicts = []
            for j, idx in enumerate(flipped_indices):
                event_ctx = get_event_context(int(events[idx]))
                verdict = llm_cross_validation(
                    text=str(orig_texts[idx]),
                    dl_pred_label=int(adv_preds[idx]),
                    dl_confidence=float(adv_confs[idx]),
                    original_pred_label=int(orig_preds[idx]),
                    event_context=event_ctx,
                    explainer=explainer
                )
                llm_verdicts.append({
                    'index': int(idx),
                    'original_text': str(orig_texts[idx]),
                    'orig_pred': int(orig_preds[idx]),
                    'adv_pred': int(adv_preds[idx]),
                    **verdict
                })

                if (j + 1) % 10 == 0:
                    print(f"  验证进度: {j+1}/{n_flipped}")

                # 速率限制保护
                import time
                time.sleep(0.5)

            print(f"  验证进度: {n_flipped}/{n_flipped}")
            print(f"  [OK] LLM 交叉验证完成")

        except ImportError as e:
            print(f"  [WARN] 无法导入 LLM 模块: {e}")
            print(f"  跳过 LLM 交叉验证")
        except ValueError as e:
            print(f"  [WARN] LLM 初始化失败（可能缺少 API key）: {e}")
            print(f"  跳过 LLM 交叉验证")
        except Exception as e:
            print(f"  [WARN] LLM 交叉验证异常: {e}")
            print(f"  跳过 LLM 交叉验证")
    else:
        print(f"\n[Step 4/5] 跳过 LLM 交叉验证（use_llm_cross_validation=False）")

    # ================================================================
    # Step 5: 脆弱模式分析 + 保存报告
    # ================================================================
    print(f"\n{'='*60}")
    print("[Step 5/5] 脆弱模式分析 + 报告生成")
    print(f"{'='*60}")

    if adv_results is not None:
        analyze_vulnerability_patterns(orig_results, adv_results, adversarial_texts, adv_output_dir)

    # 保存完整报告
    save_adversarial_report(
        output_dir=adv_output_dir,
        original_csv=original_csv,
        n_total=len(adversarial_texts),
        n_skipped=n_skipped,
        replaced_words=replaced_words_counter,
        comparison=comparison if adv_results is not None else None,
        llm_verdicts=llm_verdicts,
        use_llm=use_llm_cross_validation,
        model_path=model_path,
        adv_model_path=adv_model_path,
    )

    print(f"\n{'='*60}")
    print("全自动鲁棒性分析 — 完成")
    print(f"{'='*60}")
    print(f"  对抗样本: {adv_sample_path}")
    print(f"  推理结果: {adv_results_path}")
    print(f"  分析报告: {os.path.join(adv_output_dir, 'adversarial_report.txt')}")

    return {
        'adv_samples': adv_sample_path,
        'adv_results': adv_results_path,
        'comparison': comparison,
        'llm_verdicts': llm_verdicts,
    }


# ============================================================
# 对比分析（Phase 3.3 — 增强版）
# ============================================================

def compare_robustness(
    original_results: pd.DataFrame,
    adversarial_results: pd.DataFrame,
    output_dir: str
) -> dict:
    """
    对比原始预测和对抗样本预测（增强版）

    额外输出:
    - 翻转样本的原文 vs 对抗文本对照表（前 20 条）
    - 哪些词被替换最频繁
    - 高置信度翻转（原 confidence > 0.9 但仍被翻转）
    """
    n = min(len(original_results), len(adversarial_results))
    orig = original_results.head(n).copy()
    adv = adversarial_results.head(n).copy()

    # 统计翻转
    flipped = (orig['pred_label'].values != adv['pred_label'].values)
    flip_rate = flipped.mean()
    n_flipped = flipped.sum()

    print(f"\n  [核心指标] 攻击成功率（label 翻转率）: {flip_rate:.1%} ({n_flipped}/{n})")

    # 翻转方向
    fp_flips = ((orig['pred_label'].values == 0) & (adv['pred_label'].values == 1)).sum()
    fn_flips = ((orig['pred_label'].values == 1) & (adv['pred_label'].values == 0)).sum()
    print(f"    误报翻转 (0→1): {fp_flips}")
    print(f"    漏报翻转 (1→0): {fn_flips}")

    # 高置信度翻转
    if 'confidence' in orig.columns:
        high_conf_mask = orig['confidence'].values > 0.9
        high_conf_flipped = flipped & high_conf_mask
        n_high_flipped = high_conf_flipped.sum()
        print(f"\n  [危险] 高置信度翻转 (原confidence>0.9): {n_high_flipped} 条")

        # 列出高置信度翻转样本
        if n_high_flipped > 0:
            print(f"\n  高置信度翻转样本（前 10 条）:")
            high_flip_indices = np.where(high_conf_flipped)[0][:10]
            for i, idx in enumerate(high_flip_indices):
                text_preview = str(orig.iloc[idx].get('text', 'N/A'))[:60]
                print(f"    {i+1}. [{orig.iloc[idx]['pred_label']}→{adv.iloc[idx]['pred_label']}] "
                      f"conf={orig.iloc[idx]['confidence']:.3f} | {text_preview}...")

    # 按事件分组的翻转率
    event_comparison = {}
    if 'event' in orig.columns:
        print(f"\n  [事件维度] 各事件攻击成功率:")
        for e in sorted(orig['event'].unique()):
            mask = orig['event'].values == e
            e_flip = flipped[mask].mean()
            n_e = mask.sum()
            e_flipped_n = flipped[mask].sum()
            print(f"    Event {int(e)}: {e_flip:.1%} ({e_flipped_n}/{n_e})")
            event_comparison[int(e)] = {
                'flip_rate': e_flip,
                'n_total': int(n_e),
                'n_flipped': int(e_flipped_n)
            }

    # 翻转样本对照表（前 20 条）
    if n_flipped > 0:
        print(f"\n  [案例] 翻转样本原文 vs 对抗文本（前 10 条）:")
        flip_indices = np.where(flipped)[0][:10]
        for i, idx in enumerate(flip_indices):
            orig_text = str(orig.iloc[idx].get('text', 'N/A'))[:80]
            adv_text = str(adv.iloc[idx].get('text', 'N/A'))[:80]
            print(f"    {i+1}. [{orig.iloc[idx]['pred_label']}→{adv.iloc[idx]['pred_label']}]")
            print(f"       原文: {orig_text}")
            print(f"       对抗: {adv_text}")

    return {
        'flip_rate': flip_rate,
        'n_flipped': int(n_flipped),
        'n_total': n,
        'fp_flips': int(fp_flips),
        'fn_flips': int(fn_flips),
        'high_conf_flipped': int(n_high_flipped) if 'confidence' in orig.columns else 0,
        'event_comparison': event_comparison,
    }


# ============================================================
# 脆弱模式分析（Phase 3.4 — 增强版）
# ============================================================

def analyze_vulnerability_patterns(
    original_results: pd.DataFrame,
    adversarial_results: pd.DataFrame,
    adversarial_texts: list,
    output_dir: str
):
    """分析哪些样本更容易被攻击"""
    n = min(len(original_results), len(adversarial_results))
    orig = original_results.head(n).copy()

    # 按原置信度分组分析
    if 'confidence' in orig.columns:
        print(f"\n  [置信度维度] 原置信度与攻击成功率:")
        conf_bins = [0, 0.5, 0.7, 0.9, 1.0]
        conf_labels = ['<0.5 (低置信)', '0.5-0.7 (中置信)', '0.7-0.9 (高置信)', '>0.9 (极高置信)']
        orig['conf_bin'] = pd.cut(orig['confidence'], bins=conf_bins, labels=conf_labels)

        flipped = (orig['pred_label'].values != adversarial_results['pred_label'].values[:n])
        for label in conf_labels:
            mask = orig['conf_bin'] == label
            if mask.sum() > 0:
                bin_flip = flipped[mask].mean()
                print(f"    {label}: {bin_flip:.1%} ({mask.sum()} 条)")

    # 按预测标签分组
    print(f"\n  [标签维度] 原预测标签与攻击成功率:")
    for label_val in [0, 1]:
        label_name = "非谣言(0)" if label_val == 0 else "谣言(1)"
        mask = orig['pred_label'].values == label_val
        if mask.sum() > 0:
            label_flip = flipped[mask].mean()
            print(f"    {label_name}: {label_flip:.1%} ({mask.sum()} 条)")


# ============================================================
# 报告生成
# ============================================================

def save_adversarial_report(
    output_dir: str,
    original_csv: str,
    n_total: int,
    n_skipped: int,
    replaced_words: Counter,
    comparison: dict | None,
    llm_verdicts: list | None,
    use_llm: bool,
    model_path: str,
    adv_model_path: str | None,
):
    """保存 {output_dir}/adversarial_report.txt"""
    report_path = os.path.join(output_dir, 'adversarial_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("对抗样本攻击与防护 — 分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"[数据]\n")
        f.write(f"  原始数据: {original_csv}\n")
        f.write(f"  对抗样本总数: {n_total}\n")
        f.write(f"  可扰动: {n_total - n_skipped} 条\n")
        f.write(f"  无法扰动: {n_skipped} 条 ({n_skipped/n_total*100:.1f}%)\n\n")

        f.write(f"[模型]\n")
        f.write(f"  原始模型: {model_path}\n")
        if adv_model_path:
            f.write(f"  对抗训练模型: {adv_model_path}\n")

        # 最常被替换的词
        if replaced_words:
            f.write(f"\n[最常被替换的词 Top 10]\n")
            for word, count in replaced_words.most_common(10):
                f.write(f"  {word}: {count} 次\n")

        # 对比结果
        if comparison:
            f.write(f"\n[攻击效果]\n")
            f.write(f"  攻击成功率（label 翻转率）: {comparison['flip_rate']:.1%} "
                    f"({comparison['n_flipped']}/{comparison['n_total']})\n")
            f.write(f"  误报翻转 (0→1): {comparison['fp_flips']}\n")
            f.write(f"  漏报翻转 (1→0): {comparison['fn_flips']}\n")
            f.write(f"  高置信度翻转 (conf>0.9): {comparison.get('high_conf_flipped', 0)} 条\n")

            if comparison.get('event_comparison'):
                f.write(f"\n[各事件攻击成功率]\n")
                for eid, stats in sorted(comparison['event_comparison'].items()):
                    f.write(f"  Event {eid}: {stats['flip_rate']:.1%} "
                            f"({stats['n_flipped']}/{stats['n_total']})\n")

        # LLM 交叉验证结果
        if llm_verdicts is not None:
            f.write(f"\n[LLM 交叉验证防护]\n")
            n_llm = len(llm_verdicts)
            support = sum(1 for v in llm_verdicts if v['verdict'] == '支持DL')
            overturn = sum(1 for v in llm_verdicts if v['verdict'] == '推翻DL')
            uncertain = sum(1 for v in llm_verdicts if v['verdict'] == '不确定')

            f.write(f"  翻转样本总数: {n_llm}\n")
            f.write(f"  支持DL判断: {support} 条 (LLM 同意 DL 对抗后的判断)\n")
            f.write(f"  推翻DL判断: {overturn} 条 (LLM 纠正了对抗扰动 → 防御成功)\n")
            f.write(f"  不确定: {uncertain} 条 (建议人工复核)\n")
            defense_rate = overturn / n_llm * 100 if n_llm > 0 else 0
            f.write(f"  防御成功率: {defense_rate:.1f}%\n")

            # 列出被 LLM 推翻的案例
            overturned = [v for v in llm_verdicts if v['verdict'] == '推翻DL']
            if overturned:
                f.write(f"\n  [LLM 成功防御案例（前 10 条）]\n")
                for i, v in enumerate(overturned[:10]):
                    f.write(f"  {i+1}. [{v['orig_pred']}→{v['adv_pred']}] "
                            f"→ LLM判: {v['llm_label']}\n")
                    f.write(f"     原文: {v['original_text'][:80]}\n")
                    f.write(f"     LLM理由: {v['llm_reasoning'][:120]}\n")

        elif use_llm:
            f.write(f"\n[LLM 交叉验证]\n")
            f.write(f"  LLM 交叉验证未能执行（可能缺少 API key 或模块未正确初始化）\n")

    print(f"  [OK] 报告已保存至 {report_path}")


# ============================================================
# 四组对照实验（Phase 5）
# ============================================================

def print_defense_comparison(results: dict):
    """打印四组实验对比表"""
    print(f"""
    {'='*70}
               对抗攻击与防护 — 四组实验对比
    {'='*70}
    {'实验组':<16} {'准确率':<10} {'翻转率':<10} {'高置信翻转':<12} {'防护效果':<10}
    {'-'*70}
    {'原始（基线）':<16} {results.get('original_acc', 'N/A'):<10} {'—':<10} {'—':<12} {'—':<10}""")

    if 'A' in results:
        r = results['A']
        print(f"    {'A 无防护':<14} {r.get('acc', 'N/A'):<10} {r.get('flip_rate', 'N/A'):<10} "
              f"{r.get('high_flipped', 'N/A'):<12} {'—':<10}")

    if 'B' in results:
        r = results['B']
        flip_reduction = ''
        if results.get('A') and isinstance(results['A'].get('flip_rate'), (int, float)):
            reduction = results['A']['flip_rate'] - r.get('flip_rate', 0)
            flip_reduction = f'{reduction:.1%}↓'
        print(f"    {'B 对抗训练':<14} {r.get('acc', 'N/A'):<10} {r.get('flip_rate', 'N/A'):<10} "
              f"{r.get('high_flipped', 'N/A'):<12} {flip_reduction:<10}")

    if 'C' in results:
        r = results['C']
        flip_reduction = ''
        if results.get('A') and isinstance(results['A'].get('flip_rate'), (int, float)):
            reduction = results['A']['flip_rate'] - r.get('flip_rate', 0)
            flip_reduction = f'{reduction:.1%}↓'
        print(f"    {'C LLM交叉验证':<14} {r.get('acc', 'N/A'):<10} {r.get('flip_rate', 'N/A'):<10} "
              f"{r.get('high_flipped', 'N/A'):<12} {flip_reduction:<10}")

    if 'D' in results:
        r = results['D']
        flip_reduction = ''
        if results.get('A') and isinstance(results['A'].get('flip_rate'), (int, float)):
            reduction = results['A']['flip_rate'] - r.get('flip_rate', 0)
            flip_reduction = f'{reduction:.1%}↓'
        print(f"    {'D 双重防护':<14} {r.get('acc', 'N/A'):<10} {r.get('flip_rate', 'N/A'):<10} "
              f"{r.get('high_flipped', 'N/A'):<12} {flip_reduction:<10}")

    print(f"    {'='*70}")


def run_defense_experiments(
    original_csv: str = "rumer2026/val.csv",
    output_dir: str = "results",
    model_path: str = "checkpoints/best_model.pt",
    adv_model_path: str = None,
    device: str = "cpu",
    use_llm: bool = False,
    original_results_csv: str = None,
):
    """
    四组对照实验:

    | 实验组 | 模型 | 对抗样本 | 防护机制 | 预期翻转率 |
    |--------|------|:---:|------|:---:|
    | A 无防护 | best_model.pt | ✅ | 无 | 15-25% |
    | B 对抗训练 | 对抗训练模型 | ✅ | 训练时注入对抗样本 | 8-15% |
    | C LLM交叉验证 | best_model.pt | ✅ | LLM 对翻转样本二次判断 | 5-10% |
    | D 双重防护 | 对抗训练模型 | ✅ | 对抗训练 + LLM 交叉验证 | 3-8% |
    """
    print("=" * 70)
    print("对抗攻击与防护 — 四组对照实验")
    print("=" * 70)
    print(f"  原始数据: {original_csv}")
    print(f"  基础模型: {model_path}")
    print(f"  对抗训练模型: {adv_model_path or 'N/A (需手动提供)'}")
    print(f"  LLM 防御: {'开启' if use_llm else '关闭'}")
    print()

    all_results = {}

    # --- 获取原始准确率 ---
    if original_results_csv and os.path.exists(original_results_csv):
        orig_df = pd.read_csv(original_results_csv)
        if 'true_label' in orig_df.columns and 'pred_label' in orig_df.columns:
            orig_acc = (orig_df['true_label'] == orig_df['pred_label']).mean()
            all_results['original_acc'] = f'{orig_acc:.1%}'
    else:
        all_results['original_acc'] = 'N/A'

    # --- 实验 A：无防护 ---
    print(f"\n{'='*60}")
    print("[实验 A] 无防护模型 — 对抗样本直接攻击")
    print(f"{'='*60}")
    result_a = analyze_robustness(
        results_csv=original_results_csv,
        original_csv=original_csv,
        output_dir=os.path.join(output_dir, 'expA_no_defense'),
        model_path=model_path,
        device=device,
        auto_run_inference=True,
        use_llm_cross_validation=False,
    )
    if result_a['comparison']:
        all_results['A'] = {
            'acc': f"{1 - result_a['comparison']['flip_rate']:.1%}",
            'flip_rate': f"{result_a['comparison']['flip_rate']:.1%}",
            'high_flipped': str(result_a['comparison'].get('high_conf_flipped', 'N/A')),
        }

    # --- 实验 B：对抗训练（需要对抗训练后的模型）---
    print(f"\n{'='*60}")
    print("[实验 B] 对抗训练模型防护")
    print(f"{'='*60}")
    if adv_model_path and os.path.exists(adv_model_path):
        result_b = analyze_robustness(
            results_csv=None,
            original_csv=original_csv,
            output_dir=os.path.join(output_dir, 'expB_adv_training'),
            model_path=adv_model_path,
            device=device,
            auto_run_inference=True,
            use_llm_cross_validation=False,
            adv_model_path=adv_model_path,
        )
        if result_b['comparison']:
            all_results['B'] = {
                'acc': f"{1 - result_b['comparison']['flip_rate']:.1%}",
                'flip_rate': f"{result_b['comparison']['flip_rate']:.1%}",
                'high_flipped': str(result_b['comparison'].get('high_conf_flipped', 'N/A')),
            }
    else:
        print(f"  跳过 — 对抗训练模型不存在 ({adv_model_path or '未指定'})")
        print(f"  如需实验 B，请先运行:")
        print(f"  python train.py --adversarial --epochs 8 --lr 1e-5 --max_len 128 --batch_size 16 --save_dir checkpoints/adv_defense")
        all_results['B'] = {'acc': '待训练', 'flip_rate': '待训练', 'high_flipped': '待训练'}

    # --- 实验 C：LLM 交叉验证 ---
    print(f"\n{'='*60}")
    print("[实验 C] LLM 交叉验证防护")
    print(f"{'='*60}")
    if use_llm:
        result_c = analyze_robustness(
            results_csv=original_results_csv,
            original_csv=original_csv,
            output_dir=os.path.join(output_dir, 'expC_llm_defense'),
            model_path=model_path,
            device=device,
            auto_run_inference=True,
            use_llm_cross_validation=True,
        )
        if result_c['comparison']:
            # 计算 LLM 防御后的有效翻转率
            if result_c['llm_verdicts']:
                overturned = sum(1 for v in result_c['llm_verdicts'] if v['verdict'] == '推翻DL')
                effective_flips = result_c['comparison']['n_flipped'] - overturned
                effective_flip_rate = effective_flips / result_c['comparison']['n_total']
                all_results['C'] = {
                    'acc': f"{1 - effective_flip_rate:.1%}",
                    'flip_rate': f"{effective_flip_rate:.1%}",
                    'high_flipped': '见报告',
                }
            else:
                all_results['C'] = {
                    'acc': f"{1 - result_c['comparison']['flip_rate']:.1%}",
                    'flip_rate': f"{result_c['comparison']['flip_rate']:.1%}",
                    'high_flipped': str(result_c['comparison'].get('high_conf_flipped', 'N/A')),
                }
    else:
        print(f"  跳过 — 未启用 LLM 交叉验证")
        all_results['C'] = {'acc': '需LLM', 'flip_rate': '需LLM', 'high_flipped': '需LLM'}

    # --- 实验 D：双重防护 ---
    print(f"\n{'='*60}")
    print("[实验 D] 双重防护（对抗训练 + LLM 交叉验证）")
    print(f"{'='*60}")
    if adv_model_path and os.path.exists(adv_model_path) and use_llm:
        result_d = analyze_robustness(
            results_csv=None,
            original_csv=original_csv,
            output_dir=os.path.join(output_dir, 'expD_dual_defense'),
            model_path=adv_model_path,
            device=device,
            auto_run_inference=True,
            use_llm_cross_validation=True,
            adv_model_path=adv_model_path,
        )
        if result_d['comparison']:
            if result_d['llm_verdicts']:
                overturned = sum(1 for v in result_d['llm_verdicts'] if v['verdict'] == '推翻DL')
                effective_flips = result_d['comparison']['n_flipped'] - overturned
                effective_flip_rate = effective_flips / result_d['comparison']['n_total']
                all_results['D'] = {
                    'acc': f"{1 - effective_flip_rate:.1%}",
                    'flip_rate': f"{effective_flip_rate:.1%}",
                    'high_flipped': '见报告',
                }
            else:
                all_results['D'] = {
                    'acc': f"{1 - result_d['comparison']['flip_rate']:.1%}",
                    'flip_rate': f"{result_d['comparison']['flip_rate']:.1%}",
                    'high_flipped': str(result_d['comparison'].get('high_conf_flipped', 'N/A')),
                }
    else:
        print(f"  跳过 — 需要对抗训练模型 + LLM 交叉验证")
        all_results['D'] = {'acc': '需两者', 'flip_rate': '需两者', 'high_flipped': '需两者'}

    # --- 打印对比表 ---
    print(f"\n")
    print_defense_comparison(all_results)

    # --- 保存对比表到文件 ---
    comparison_path = os.path.join(output_dir, 'defense_comparison.txt')
    os.makedirs(output_dir, exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("对抗攻击与防护 — 四组实验对比\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for exp_name in ['original', 'A', 'B', 'C', 'D']:
            exp_key = 'original_acc' if exp_name == 'original' else exp_name
            if exp_key in all_results:
                r = all_results[exp_key]
                if exp_name == 'original':
                    f.write(f"原始准确率: {r}\n")
                else:
                    f.write(f"实验{exp_name}: acc={r.get('acc')}, flip_rate={r.get('flip_rate')}, "
                            f"high_flipped={r.get('high_flipped')}\n")
        f.write(f"\n模型路径: {model_path}\n")
        f.write(f"对抗训练模型: {adv_model_path or 'N/A'}\n")
    print(f"\n  对比表已保存至 {comparison_path}")

    return all_results


# ============================================================
# CLI 入口（Phase 6）
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="对抗样本攻击与防护 — 全自动攻防评估管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整攻防评估（一条命令跑完）
  python adversarial.py --mode full --original rumer2026/val.csv --use-llm-defense

  # 仅生成对抗样本
  python adversarial.py --mode generate --original rumer2026/val.csv

  # 仅评估（需要已有对抗样本推理结果）
  python adversarial.py --mode evaluate \\
      --input results/val_results.csv \\
      --adversarial results/adversarial_results.csv

  # 四组对照实验
  python adversarial.py --mode experiments \\
      --original rumer2026/val.csv \\
      --adv-model checkpoints/adv_defense/best_model.pt \\
      --use-llm-defense
        """
    )

    # --- 模式 ---
    parser.add_argument('--mode', default='full',
                        choices=['full', 'generate', 'evaluate', 'experiments'],
                        help='运行模式: full=全自动攻防评估, generate=仅生成对抗样本, '
                             'evaluate=仅评估对比, experiments=四组对照实验')

    # --- 输入 ---
    parser.add_argument('--input', default=None,
                        help='原始验证集推理结果 CSV (inference.py 输出)')
    parser.add_argument('--original', default='rumer2026/val.csv',
                        help='原始验证集 CSV')
    parser.add_argument('--adversarial', default=None,
                        help='对抗样本推理结果 CSV（mode=evaluate 时需要）')

    # --- 模型 ---
    parser.add_argument('--model', default='checkpoints/best_model.pt',
                        help='基础模型路径')
    parser.add_argument('--adv-model', default=None,
                        help='对抗训练后模型路径（实验 B/D 需要）')

    # --- 输出 ---
    parser.add_argument('--output-dir', default='results',
                        help='输出根目录')

    # --- 选项 ---
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda', 'auto'],
                        help='推理设备')
    parser.add_argument('--use-llm-defense', action='store_true',
                        help='启用 LLM 交叉验证防护')
    parser.add_argument('--no-auto-inference', action='store_true',
                        help='禁用自动推理（使用已有结果）')

    args = parser.parse_args()

    # --- 执行 ---
    if args.mode == 'generate':
        # 仅生成对抗样本
        print("=" * 60)
        print("对抗样本生成")
        print("=" * 60)
        original = pd.read_csv(args.original)
        adv_texts = []
        n_skipped = 0
        for _, row in original.iterrows():
            orig_text = str(row['text'])
            adv_text = generate_adversarial(orig_text)
            if adv_text == orig_text:
                n_skipped += 1
            adv_texts.append(adv_text)

        adv_df = original.copy()
        adv_df['text'] = adv_texts

        adv_dir = os.path.join(args.output_dir, 'adversarial')
        os.makedirs(adv_dir, exist_ok=True)
        adv_path = os.path.join(adv_dir, 'adversarial_samples.csv')
        adv_df.to_csv(adv_path, index=False)
        print(f"  生成 {len(adv_texts)} 条对抗样本 ({n_skipped} 条无法扰动)")
        print(f"  [OK] 已保存至 {adv_path}")

    elif args.mode == 'evaluate':
        # 仅评估
        if not args.input:
            print("[ERROR] --mode evaluate 需要 --input（原始推理结果）")
            sys.exit(1)
        if not args.adversarial:
            print("[ERROR] --mode evaluate 需要 --adversarial（对抗样本推理结果）")
            sys.exit(1)

        orig_results = pd.read_csv(args.input)
        adv_results = pd.read_csv(args.adversarial)
        compare_robustness(orig_results, adv_results, os.path.join(args.output_dir, 'adversarial'))

    elif args.mode == 'experiments':
        # 四组对照实验
        run_defense_experiments(
            original_csv=args.original,
            output_dir=args.output_dir,
            model_path=args.model,
            adv_model_path=args.adv_model,
            device=args.device,
            use_llm=args.use_llm_defense,
            original_results_csv=args.input,
        )

    else:  # mode == 'full'
        # 全自动攻防评估
        analyze_robustness(
            results_csv=args.input,
            original_csv=args.original,
            output_dir=args.output_dir,
            model_path=args.model,
            device=args.device,
            auto_run_inference=not args.no_auto_inference,
            use_llm_cross_validation=args.use_llm_defense,
            adv_model_path=args.adv_model,
        )


if __name__ == '__main__':
    main()
