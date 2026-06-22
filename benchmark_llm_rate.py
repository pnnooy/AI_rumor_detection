"""
LLM 速率 Benchmark — 寻找最佳线程数/间隔组合

SJTU API 官方限制: 10 RPM (每分钟10次请求)

用法:
    python benchmark_llm_rate.py
    python benchmark_llm_rate.py --n-samples 20
"""

import time
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from llm_explainer import LLMExplainer
from event_context import EVENT_CONTEXT

# ============================================================
# 配置
# ============================================================

# 测试参数组合
CONFIGS = [
    # (线程数, 调用间隔秒)
    (1, 6.0),    # 保守: 1线程, 6s → 刚好10 RPM
    (1, 5.0),    # 略激进: 1线程, 5s → 12 RPM
    (2, 7.0),    # 2线程, 7s间隔
    (2, 6.0),    # 2线程, 6s间隔
    (3, 8.0),    # 3线程, 8s间隔
    (3, 7.0),    # 3线程, 7s间隔
]

# 每个配置测试的样本数
DEFAULT_N_SAMPLES = 15


def run_benchmark(n_workers: int, min_interval: float, items: list, explainer: LLMExplainer):
    """测试一组参数配置"""
    n_total = len(items)
    completed = [0]
    failed = [0]
    print_lock = Lock()
    rate_lock = Lock()
    last_call_time = [0.0]  # 初始化为0，让第一个请求立即发出

    def call_llm(item):
        # 速率控制
        with rate_lock:
            now = time.time()
            wait = min_interval - (now - last_call_time[0])
            if wait > 0 and last_call_time[0] > 0:
                time.sleep(wait)
            last_call_time[0] = time.time()

        try:
            explanation = explainer.explain(
                text=item['text'],
                dl_result=item['dl_result'],
                similar_cases=[],
                event_info=item.get('event_info', '')
            )
            ok = not explanation.startswith('解释生成失败') and not explanation.startswith('[LLM')
        except Exception as e:
            explanation = str(e)
            ok = False

        with print_lock:
            completed[0] += 1
            if not ok:
                failed[0] += 1
            elapsed = time.time() - t_start
            rpm = completed[0] / (elapsed / 60) if elapsed > 0 else 0
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} [{completed[0]:>3}/{n_total}] {rpm:.1f} RPM | "
                  f"{item['text'][:50]}...", flush=True)

    t_start = time.time()
    print(f"  开始测试: {n_workers}线程, {min_interval}s间隔, {n_total}条样本")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(call_llm, item) for item in items]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - t_start
    rpm = n_total / (elapsed / 60)
    success_rate = (n_total - failed[0]) / n_total * 100

    return {
        'n_workers': n_workers,
        'min_interval': min_interval,
        'n_samples': n_total,
        'elapsed_s': elapsed,
        'rpm': rpm,
        'failed': failed[0],
        'success_rate': success_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM 速率 Benchmark")
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES,
                        help=f'每个配置测试的样本数 (默认: {DEFAULT_N_SAMPLES})')
    args = parser.parse_args()

    # 加载数据
    print("=" * 60)
    print("LLM 速率 Benchmark")
    print("=" * 60)

    df = pd.read_csv('rumer2026/val.csv')
    print(f"加载验证集: {len(df)} 条")

    # 初始化 explainer
    print("初始化 LLMExplainer...")
    explainer = LLMExplainer(max_retries=1)  # benchmark 不重试，直接测成功率
    print(f"模型: {explainer.model}")

    # 准备测试数据（取前 N 条，尽量覆盖不同事件）
    test_df = df.head(args.n_samples)
    test_items = []
    for _, row in test_df.iterrows():
        text = str(row['text'])
        event_id = int(row['event'])
        test_items.append({
            'text': text,
            'dl_result': {'label': 0, 'confidence': 0.5, 'keywords': []},
            'event_info': EVENT_CONTEXT.get(event_id, ""),
            'similar_cases': [],
        })

    print(f"\n测试样本: {len(test_items)} 条")
    print(f"测试配置: {len(CONFIGS)} 组\n")

    # 跑 benchmark
    results = []
    for i, (n_workers, interval) in enumerate(CONFIGS):
        print(f"\n--- 配置 {i+1}/{len(CONFIGS)} ---")
        # 每个配置跑之前休息一下，清空 API 限流窗口
        if i > 0:
            print("  等待60秒清空限流窗口...")
            time.sleep(60)

        result = run_benchmark(n_workers, interval, test_items, explainer)
        results.append(result)
        print(f"  结果: {result['elapsed_s']:.0f}s, "
              f"{result['rpm']:.1f} RPM, "
              f"成功率 {result['success_rate']:.1f}% "
              f"(失败{result['failed']}条)")

    # 汇总
    print("\n" + "=" * 70)
    print("Benchmark 汇总 (官方限制: 10 RPM)")
    print("=" * 70)
    print(f"{'线程':<6} {'间隔':<8} {'耗时':<8} {'实际RPM':<10} {'成功率':<10}")
    print("-" * 50)
    for r in results:
        marker = " ⭐" if r['success_rate'] == 100 and r['failed'] == 0 else ""
        print(f"{r['n_workers']:<6} {r['min_interval']:<6.0f}s  "
              f"{r['elapsed_s']:<6.0f}s  {r['rpm']:<8.1f}  "
              f"{r['success_rate']:<8.1f}%{marker}")
    print("-" * 50)

    # 推荐
    perfect = [r for r in results if r['failed'] == 0]
    if perfect:
        best = max(perfect, key=lambda r: r['rpm'])
        print(f"\n推荐配置: {best['n_workers']}线程, {best['min_interval']:.0f}s间隔 "
              f"(实测 {best['rpm']:.1f} RPM, 成功率100%)")
    else:
        best = max(results, key=lambda r: r['success_rate'])
        print(f"\n最佳配置: {best['n_workers']}线程, {best['min_interval']:.0f}s间隔 "
              f"(成功率 {best['success_rate']:.1f}%)")

    # 预估 401 条耗时
    if perfect:
        r = best
        estimated = 401 / (r['rpm'] / 60)
        print(f"预估401条总耗时: {estimated/60:.1f} 分钟")


if __name__ == '__main__':
    main()
