# tests/test_retrieval.py
# 检索模块测试脚本
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import CaseRetriever, get_retriever


def test_build_index():
    """测试索引构建"""
    print("=== 测试索引构建 ===")
    retriever = CaseRetriever()
    
    # 检查数据文件是否存在
    if not os.path.exists('rumer2026/train.csv'):
        print("警告: 训练数据不存在，跳过索引构建测试")
        return
    
    retriever.build_index('rumer2026/train.csv')
    print(f"索引构建完成，包含 {len(retriever.texts)} 条推文")
    assert retriever.embeddings is not None
    assert len(retriever.texts) == len(retriever.labels)
    print("✓ 索引构建测试通过\n")


def test_load_index():
    """测试索引加载"""
    print("=== 测试索引加载 ===")
    retriever = CaseRetriever()
    
    result = retriever.load_index()
    if result:
        print(f"索引加载成功，包含 {len(retriever.texts)} 条推文")
        assert retriever.embeddings is not None
        print("✓ 索引加载测试通过\n")
    else:
        print("跳过: 索引文件不存在\n")


def test_search():
    """测试检索功能"""
    print("=== 测试检索功能 ===")
    retriever = get_retriever()
    
    test_queries = [
        "Police fired tear gas at protesters",
        "Breaking news about the election results",
        "Scientists discover new treatment for disease"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, top_k=3)
        for i, case in enumerate(results):
            label_str = "谣言" if case['label'] == 1 else "非谣言"
            print(f"  {i+1}. [相似度: {case['similarity']:.3f}] {label_str}")
            print(f"     {case['text'][:80]}...")
        
        # 验证结果数量
        assert len(results) == 3
        print("✓ 检索返回数量正确")
        
        # 验证结果包含必要字段
        for case in results:
            assert 'text' in case
            assert 'label' in case
            assert 'event' in case
            assert 'similarity' in case
        print("✓ 结果包含所有必要字段")


def test_search_with_diversity():
    """测试多样性检索"""
    print("\n=== 测试多样性检索 ===")
    retriever = get_retriever()
    
    query = "Police fired tear gas at protesters"
    results = retriever.search_with_diversity(query, top_k=3)
    
    # 检查标签多样性
    labels = [case['label'] for case in results]
    if len(set(labels)) > 1:
        print("✓ 检索结果包含多种标签")
    else:
        print("! 检索结果标签单一（可能数据集中该查询的相似案例标签单一）")
    
    # 打印结果
    for i, case in enumerate(results):
        label_str = "谣言" if case['label'] == 1 else "非谣言"
        print(f"  {i+1}. [相似度: {case['similarity']:.3f}] {label_str}")
        print(f"     {case['text'][:80]}...")


if __name__ == '__main__':
    print("运行检索模块测试...\n")
    
    try:
        test_build_index()
        test_load_index()
        test_search()
        test_search_with_diversity()
        print("\n=== 所有测试通过 ===")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()