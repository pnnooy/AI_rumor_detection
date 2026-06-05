"""
端到端推理管道 — 韩宇飞 实现

用法:
    python inference.py --input rumer2026/val.csv --output results/val_results.csv
    python inference.py --input rumer2026/val.csv --output results/val_results.csv --no-llm
"""

import argparse
import sys
import pandas as pd
import torch
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 接口定义 — 姜新晨 实现
# ============================================================

def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    姜新晨 实现：加载分类模型和 tokenizer

    Args:
        checkpoint_path: 模型权重文件路径

    Returns:
        (model, tokenizer) 元组
    """
    raise NotImplementedError("姜新晨 实现 — 加载 BERT 模型和 tokenizer")


def predict(text: str, event_id: int) -> dict:
    """
    姜新晨 实现：对单条推文做分类预测 + 关键词提取

    Args:
        text:     原推文文本（未经清洗）
        event_id: 事件 ID (0-6)

    Returns:
        {
            "label":      0 或 1,
            "confidence": float (0.0 ~ 1.0),
            "keywords":   [("word", 0.23), ("word2", 0.16), ...]  # top-5
        }
    """
    raise NotImplementedError("姜新晨 实现 — 分类推理 + attention 关键词提取")


# ============================================================
# 接口定义 — 靳卓达 实现
# ============================================================

class CaseRetriever:
    """
    靳卓达 实现：相似案例检索器
    """
    def load_index(self, index_path: str = "data/index.pkl"):
        """加载训练集向量索引"""
        raise NotImplementedError("靳卓达 实现 — 加载 sentence-transformer 索引")

    def search(self, query_text: str, top_k: int = 3) -> list:
        """
        检索 top_k 条最相似训练集推文

        Returns:
            [
                {"text": str, "label": int, "event": int, "similarity": float},
                ...
            ]
        """
        raise NotImplementedError("靳卓达 实现 — 余弦相似度检索")


def generate_explanation(
    text: str,
    dl_result: dict,
    cases: list,
    event_context: str
) -> str:
    """
    靳卓达 实现：调用 LLM 生成中文解释

    Args:
        text:          原推文文本
        dl_result:     predict() 的返回结果
        cases:         CaseRetriever.search() 的返回结果
        event_context: 事件背景文本（来自 event_context.py）

    Returns:
        中文解释字符串 (150-300字)
    """
    raise NotImplementedError("靳卓达 实现 — LLM 提示词 + API 调用")


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
    # 0. 检查 API key
    if use_llm and not os.getenv("SJTU_API_KEY"):
        print("⚠️  警告: 未设置 SJTU_API_KEY，自动切换为 --no-llm 模式")
        print("   设置方法: 复制 .env.example 为 .env，填入 API key")
        use_llm = False

    # 1. 加载模型
    print("[1/4] 加载 DL 分类模型...")
    try:
        model, tokenizer = load_model(model_path)
        print("  ✓ 模型加载成功")
    except NotImplementedError:
        print("  ✗ 模型尚未实现（姜新晨 待完成），使用占位模式继续")
        model, tokenizer = None, None

    # 2. 加载检索器
    retriever = None
    if use_llm:
        print("[2/4] 加载案例检索索引...")
        try:
            retriever = CaseRetriever()
            retriever.load_index(index_path)
            print("  ✓ 索引加载成功")
        except NotImplementedError:
            print("  ✗ 检索模块尚未实现（靳卓达 待完成），将不使用相似案例")
            retriever = None
    else:
        print("[2/4] --no-llm 模式，跳过检索")

    # 3. 读取数据
    print(f"[3/4] 读取输入: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"  共 {total} 条推文")

    # 4. 逐条推理
    print(f"[4/4] 开始推理{' (含 LLM 解释)' if use_llm else ' (仅分类)'}...")

    from event_context import EVENT_CONTEXT

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row['text'])
        event_id = int(row['event'])

        # DL 分类
        try:
            dl_result = predict(text, event_id)
        except NotImplementedError:
            # 占位：在 姜新晨 实现之前使用假结果
            dl_result = {"label": 0, "confidence": 0.5, "keywords": [("待实现", 0.0)]}

        # 检索相似案例
        cases = []
        if retriever is not None:
            try:
                cases = retriever.search(text, top_k=3)
            except Exception:
                pass

        # LLM 解释
        explanation = ""
        if use_llm:
            try:
                explanation = generate_explanation(
                    text, dl_result, cases,
                    EVENT_CONTEXT.get(event_id, "")
                )
            except NotImplementedError:
                explanation = "[待实现 — 靳卓达]"
            except Exception as e:
                explanation = f"[LLM 调用失败: {e}]"

        results.append({
            'id':           row['id'],
            'text':         text,
            'true_label':   row['label'],
            'pred_label':   dl_result['label'],
            'confidence':   dl_result['confidence'],
            'keywords':     ','.join([w for w, _ in dl_result['keywords']]),
            'explanation':  explanation
        })

        # 进度
        if (i + 1) % 50 == 0 or i + 1 == total:
            pct = (i + 1) / total * 100
            print(f"  进度: {i+1}/{total} ({pct:.0f}%)")

    # 5. 保存结果
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 统计
    correct = sum(1 for r in results if r['true_label'] == r['pred_label'])
    acc = correct / len(results) * 100
    print(f"\n✓ 完成! 准确率: {correct}/{len(results)} = {acc:.1f}%")
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

    run_inference(
        input_csv=args.input,
        output_csv=args.output,
        use_llm=not args.no_llm,
        model_path=args.model,
        index_path=args.index
    )


if __name__ == '__main__':
    main()
