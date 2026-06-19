"""
端到端推理管道 — 韩宇飞 实现

用法:
    python inference.py --input rumer2026/val.csv --output results/val_results.csv
    python inference.py --input rumer2026/val.csv --output results/val_results.csv --no-llm
"""

# 立即输出，避免导入阶段用户以为程序卡死
import sys
print("正在初始化... (加载依赖库)", flush=True)

import argparse
import time
import pandas as pd
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

load_dotenv()
print("依赖库就绪", flush=True)


# ============================================================
# 接口 — 姜新晨 实现
# ============================================================

from models.classifier import load_model      # noqa: E402
from models.keyword_extractor import predict  # noqa: E402


# ============================================================
# 接口 — 靳卓达 实现
# ============================================================

from retrieval import CaseRetriever                                   # noqa: E402
from llm_explainer import LLMExplainer                                # noqa: E402


def generate_explanation(
    text: str,
    dl_result: dict,
    cases: list,
    event_context: str,
    explainer: "LLMExplainer | None" = None
) -> str:
    """
    薄封装：调用 LLMExplainer 生成中文解释

    Args:
        text:          原推文文本
        dl_result:     predict() 的返回结果
        cases:         CaseRetriever.search() 的返回结果
        event_context: 事件背景文本（来自 event_context.py）
        explainer:     共享的 LLMExplainer 实例（复用避免重复创建）

    Returns:
        中文解释字符串
    """
    if explainer is None:
        explainer = LLMExplainer()
    return explainer.explain(text, dl_result, similar_cases=cases, event_info=event_context)


# ============================================================
# 韩宇飞 实现：主推理管道
# ============================================================

def run_inference(
    input_csv: str,
    output_csv: str,
    use_llm: bool = True,
    model_path: str = "checkpoints/best_model.pt",
    index_path: str = "data/index.pkl"
):
    """
    端到端推理管道：串联 姜新晨 的 DL 分类 + 靳卓达 的检索和 LLM 解释

    Args:
        input_csv:  输入 CSV（格式同 val.csv）
        output_csv: 输出 CSV（包含预测 + 解释）
        use_llm:    False 则跳过 LLM 调用，仅输出 DL 分类结果
        model_path: 模型权重路径
        index_path: 检索索引路径
    """
    import sys

    # 0. 检查 API key
    print("=" * 60)
    print("  可解释谣言检测 — 端到端推理")
    print("=" * 60)
    if use_llm and not os.getenv("SJTU_API_KEY"):
        print("[WARN] 未设置 SJTU_API_KEY，自动切换为 --no-llm 模式")
        print("   设置方法: 复制 .env.example 为 .env，填入 API key")
        use_llm = False

    # 1. 加载模型
    print("[1/4] 加载 DL 分类模型...", end=" ", flush=True)
    print("(RoBERTa-large ~1.3GB, 约30秒)...", flush=True)
    model, tokenizer = load_model(model_path, device="cpu")
    print("  [OK] 模型加载成功")

    # 2. 加载检索器 + 初始化 LLM Explainer
    retriever = None
    explainer = None
    if use_llm:
        print("[2/4] 加载案例检索索引 (sentence-transformers ~90MB)...", flush=True)
        retriever = CaseRetriever()
        if retriever.load_index(index_path):
            print("  [OK] 索引加载成功 (2840 条训练集向量)")
        else:
            print("  [WARN] 索引文件不存在，将不使用相似案例")
            retriever = None

        print("  初始化 LLM Explainer (SJTU API: deepseek-chat)...", end=" ", flush=True)
        explainer = LLMExplainer()
        print("[OK]")
        print(f"  模型={explainer.model}, 速率控制=0.6s/req (~100 req/min)")
    else:
        print("[2/4] --no-llm 模式，跳过检索和 LLM")

    # 3. 读取数据
    print(f"[3/4] 读取输入: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"  共 {total} 条推文")

    # 4. 推理
    print(f"[4/4] 开始推理{' (含 LLM 验证, 3线程并行)' if use_llm else ' (仅分类)'}...", flush=True)

    from event_context import EVENT_CONTEXT

    start_time = time.time()

    # ---------- Phase A: DL分类 + 检索 (串行, 本地快) ----------
    print("  [Phase A] DL分类 + 案例检索...", end=" ", flush=True)
    items = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row['text'])
        event_id = int(row['event'])
        dl_result = predict(text, event_id)
        cases = []
        if retriever is not None:
            try:
                cases = retriever.search(text, top_k=3)
            except Exception:
                pass
        items.append({
            'idx':  i,
            'id':   row['id'],
            'text': text,
            'event': event_id,
            'true_label': row['label'],
            'pred_label': dl_result['label'],
            'confidence': dl_result['confidence'],
            'keywords': ','.join([w for w, _ in dl_result['keywords']]),
            'dl_result': dl_result,
            'cases': cases,
            'explanation': '',
        })
    phase_a_time = time.time() - start_time
    print(f"完成 ({phase_a_time:.0f}s)", flush=True)

    # ---------- Phase B: LLM 解释 (并行, 3线程) ----------
    if use_llm:
        n_workers = 3
        n_total = len(items)
        completed = [0]
        print_lock = Lock()
        rate_lock = Lock()
        last_call_time = [time.time()]
        min_interval = 0.6

        def call_llm(item):
            # 速率控制
            with rate_lock:
                now = time.time()
                wait = min_interval - (now - last_call_time[0])
                if wait > 0:
                    time.sleep(wait)
                last_call_time[0] = time.time()

            try:
                explanation = generate_explanation(
                    item['text'], item['dl_result'], item['cases'],
                    EVENT_CONTEXT.get(item['event'], ""),
                    explainer=explainer
                )
            except Exception as e:
                explanation = f"[LLM 调用失败: {e}]"
            item['explanation'] = explanation

            with print_lock:
                completed[0] += 1
                elapsed = time.time() - start_time
                avg_per = elapsed / completed[0]
                remaining = avg_per * (n_total - completed[0])
                eta = f"{remaining/60:.1f}min" if remaining > 60 else f"{remaining:.0f}s"
                i = item['idx']
                print(f"[{completed}/{n_total} eta={eta}] pred={item['pred_label']} conf={item['confidence']:.4f} true={item['true_label']} | {item['text'][:60]}")
                if explanation:
                    print(f"        LLM: {explanation}")
                print("-" * 60)

        print(f"  [Phase B] LLM 解释 (并行 x{n_workers})...", flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(call_llm, item) for item in items]
            for f in as_completed(futures):
                f.result()  # 等待完成, 打印已在 call_llm 中完成
    else:
        for item in items:
            elapsed = time.time() - start_time
            avg_per = elapsed / (item['idx'] + 1)
            remaining = avg_per * (len(items) - item['idx'] - 1)
            eta = f"{remaining/60:.1f}min" if remaining > 60 else f"{remaining:.0f}s"
            i = item['idx']
            print(f"[{i+1}/{total} eta={eta}] pred={item['pred_label']} conf={item['confidence']:.4f} true={item['true_label']} | {item['text'][:60]}")
            print("-" * 60)

    # 还原为 results 列表
    items.sort(key=lambda x: x['idx'])
    results = [{
        'id': it['id'], 'text': it['text'], 'event': it['event'],
        'true_label': it['true_label'], 'pred_label': it['pred_label'],
        'confidence': it['confidence'], 'keywords': it['keywords'],
        'explanation': it['explanation'],
    } for it in items]

    # 5. 保存结果
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 统计
    total_time = time.time() - start_time
    correct = sum(1 for r in results if r['true_label'] == r['pred_label'])
    acc = correct / len(results) * 100
    print(f"\n[OK] 完成! 总耗时: {total_time/60:.1f} min")
    print(f"  准确率: {correct}/{len(results)} = {acc:.1f}%")
    print(f"  结果已保存至: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="可解释谣言检测 — 端到端推理管道"
    )
    parser.add_argument(
        '--input', required=True,
        help='输入 CSV 文件路径 (如 rumer2026/val.csv)'
    )
    parser.add_argument(
        '--output', required=True,
        help='输出 CSV 文件路径 (如 results/val_results.csv)'
    )
    parser.add_argument(
        '--no-llm', action='store_true',
        help='跳过 LLM 调用，仅输出 DL 分类结果（无需 API key）'
    )
    parser.add_argument(
        '--model', default='checkpoints/best_model.pt',
        help='模型权重路径 (默认: checkpoints/best_model.pt)'
    )
    parser.add_argument(
        '--index', default='data/index.pkl',
        help='检索索引路径 (默认: data/index.pkl)'
    )
    args = parser.parse_args()
    sys.stdout.flush()

    run_inference(
        input_csv=args.input,
        output_csv=args.output,
        use_llm=not args.no_llm,
        model_path=args.model,
        index_path=args.index
    )


if __name__ == '__main__':
    main()
