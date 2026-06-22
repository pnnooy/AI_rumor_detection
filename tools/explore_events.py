# explore_events.py — 查看各事件数据分布
import pandas as pd


def main():
    df = pd.read_csv('rumer2026/train.csv')
    for event_id in range(7):
        event_data = df[df['event'] == event_id]
        print(f"\n{'='*60}")
        print(f"Event {event_id}: 共 {len(event_data)} 条推文")
        print(f"谣言占比: {event_data['label'].mean():.1%}")
        print(f"\n前5条推文示例:")
        for i, (idx, row) in enumerate(event_data.head(5).iterrows()):
            label = "谣言" if row['label'] == 1 else "非谣言"
            print(f"  {i+1}. [{label}] {row['text'][:100]}...")


if __name__ == '__main__':
    main()
