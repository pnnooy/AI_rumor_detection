# tests/test_llm_explainer.py
# LLM 解释模块测试脚本
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_explainer import LLMExplainer, get_explainer, quick_explain
from retrieval import get_retriever
from event_context import get_event_context


def test_prompt_building():
    """测试提示词构建"""
    print("=== 测试提示词构建 ===")
    explainer = get_explainer()
    
    text = "Witness says police fired tear gas at protesters"
    dl_result = {
        'label': 1,
        'confidence': 0.87,
        'keywords': [('witness', 0.23), ('says', 0.16), ('fired', 0.11)]
    }
    similar_cases = [
        {'text': 'Eyewitness claims police used excessive force', 
         'label': 1, 'event': 5, 'similarity': 0.89},
        {'text': 'Official statement confirms police followed protocol',
         'label': 0, 'event': 5, 'similarity': 0.78}
    ]
    event_info = "关于 Ferguson 事件的讨论"
    
    prompt = explainer.build_prompt(text, dl_result, similar_cases, event_info)
    
    # 验证提示词包含必要元素
    assert "谣言" in prompt or "非谣言" in prompt
    assert "置信度" in prompt
    assert "关键词" in prompt
    assert "相似案例" in prompt or "案例" in prompt
    assert "事件背景" in prompt
    
    print("提示词构建成功:")
    print("-" * 50)
    print(prompt[:500] + "...\n")
    print("✓ 提示词包含所有必要要素\n")


def test_real_api_call():
    """测试真实 API 调用（需要配置 API Key）"""
    print("=== 测试真实 API 调用 ===")
    
    try:
        explainer = get_explainer()
        
        text = "Witness says police fired tear gas at protesters"
        dl_result = {
            'label': 1,
            'confidence': 0.87,
            'keywords': [('witness', 0.23), ('says', 0.16), ('fired', 0.11)]
        }
        similar_cases = [
            {'text': 'Eyewitness claims police used excessive force', 
             'label': 1, 'event': 5, 'similarity': 0.89},
            {'text': 'Official statement confirms police followed protocol',
             'label': 0, 'event': 5, 'similarity': 0.78}
        ]
        event_info = get_event_context(1)
        
        print("正在调用 API 生成解释...")
        explanation = explainer.explain(text, dl_result, similar_cases, event_info)
        
        print("生成的解释:")
        print("-" * 50)
        print(explanation)
        print("-" * 50)
        
        # 验证解释
        assert len(explanation) > 20
        assert "谣言" in explanation or "非谣言" in explanation
        assert "置信度" in explanation or "判断" in explanation
        
        print(f"解释长度: {len(explanation)} 字符")
        print(f"统计信息: {explainer.get_stats()}")
        print("✓ API 调用测试通过\n")
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请先配置 .env 文件中的 SJTU_API_KEY")
    except Exception as e:
        print(f"API 调用失败: {e}")
        print("请检查网络连接和 API Key 是否有效")


def test_integration_with_retrieval():
    """测试与检索模块的集成"""
    print("=== 测试与检索模块集成 ===")
    
    try:
        # 获取检索器
        retriever = get_retriever()
        
        # 测试文本
        text = "Police fired tear gas at protesters"
        
        # 检索相似案例
        similar_cases = retriever.search_with_diversity(text, top_k=3)
        print(f"检索到 {len(similar_cases)} 个相似案例")
        
        # 模拟 DL 结果
        dl_result = {
            'label': 1,
            'confidence': 0.87,
            'keywords': [('witness', 0.23), ('says', 0.16)]
        }
        
        # 获取事件背景
        event_info = get_event_context(1)
        
        # 生成解释
        explainer = get_explainer()
        explanation = explainer.explain(text, dl_result, similar_cases, event_info)
        
        print("生成的集成解释:")
        print("-" * 50)
        print(explanation)
        print("-" * 50)
        print("✓ 集成测试通过\n")
        
    except Exception as e:
        print(f"集成测试失败: {e}")


if __name__ == '__main__':
    print("运行 LLM 解释模块测试...\n")
    
    test_prompt_building()
    test_real_api_call()
    test_integration_with_retrieval()
    
    print("\n=== 测试完成 ===")