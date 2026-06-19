# llm_explainer.py
# LLM 解释生成模块 - 通过 SJTU API 调用 DeepSeek-V3.2
import os
import time
from typing import Dict, List, Optional
from openai import OpenAI
# from dotenv import load_dotenv


# 加载环境变量
# load_dotenv()


class LLMExplainer:
    """
    LLM 解释生成器：通过 SJTU API 调用 DeepSeek-V3.2
    生成自然语言解释，帮助用户理解谣言检测的判断依据。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://models.sjtu.edu.cn/api/v1",
        model: str = "deepseek-chat",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化解释器
        
        Args:
            api_key: SJTU API Key，如果不提供则从环境变量读取
            base_url: API 基础 URL
            model: 使用的模型名称
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.api_key = api_key or os.getenv("SJTU_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未找到 API Key，请在 .env 文件中设置 SJTU_API_KEY，"
                "或通过 api_key 参数传入"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 统计信息
        self.total_calls = 0
        self.total_tokens = 0
        
    def _get_confidence_level(self, confidence: float) -> str:
        """
        根据置信度返回分级描述
        
        Args:
            confidence: 置信度 (0-1)
            
        Returns:
            str: 置信度分级描述
        """
        if confidence > 0.9:
            return "高度确信"
        elif confidence > 0.7:
            return "倾向于判定"
        else:
            return "勉强判定，建议人工复核"
    
    def _format_cases(self, similar_cases: List[Dict]) -> str:
        """
        格式化相似案例供提示词使用
        
        Args:
            similar_cases: 相似案例列表
            
        Returns:
            str: 格式化后的案例文本
        """
        if not similar_cases:
            return "（无相似案例可供参考）"
        
        lines = []
        for i, case in enumerate(similar_cases):
            label_str = "谣言" if case['label'] == 1 else "非谣言"
            lines.append(
                f"案例{i+1}: \"{case['text']}\"\n"
                f"  → 真实标签: {label_str} | 相似度: {case['similarity']:.2f}"
            )
        return "\n".join(lines)
    
    def build_prompt(
        self,
        text: str,
        dl_result: Dict,
        similar_cases: List[Dict],
        event_info: str = ""
    ) -> str:
        """
        构建包含五要素的提示词 —— LLM 作为 DL 判断的解释者

        Args:
            text: 原始推文文本
            dl_result: DL 分类器结果，包含 label, confidence, keywords
            similar_cases: 相似案例列表
            event_info: 事件背景信息

        Returns:
            str: 完整的提示词
        """
        label_str = "谣言" if dl_result['label'] == 1 else "非谣言"
        confidence = dl_result.get('confidence', 0.5)
        keywords = [w for w, _ in dl_result.get('keywords', [])]
        conf_desc = self._get_confidence_level(confidence)

        keywords_str = ', '.join(keywords) if keywords else "（无显著关键词）"
        cases_str = self._format_cases(similar_cases)
        event_str = event_info if event_info else "（无具体事件背景信息）"

        prompt = f"""你是一个谣言检测系统的解释模块。系统已经对一条社交平台推文做出了自动判断，你需要帮助用户理解判断依据。

[推文内容]
"{text}"

[事件背景]
{event_str}

[模型判断结果]
判定: {label_str}
置信度: {confidence:.0%}
系统对此判断{conf_desc}

[模型关注的关键词（按重要性排序）]
{keywords_str}

[训练集中最相似的案例]
{cases_str}

请输出中文解释（200字以内），包含以下要点：
1. 这条推文为什么被判定为{label_str}？结合关键词和文本线索分析
2. 判断的可信度如何？如果置信度较低，应坦诚说明不确定性
3. 有什么需要人工复核的地方吗？

请直接输出解释文本，不要包含标题或前缀。"""

        return prompt

    def explain(
        self,
        text: str,
        dl_result: Dict,
        similar_cases: List[Dict],
        event_info: str = ""
    ) -> str:
        """
        生成解释

        Args:
            text: 原始推文文本
            dl_result: DL 分类器结果
            similar_cases: 相似案例列表
            event_info: 事件背景信息

        Returns:
            str: 自然语言解释文本（中文）
        """
        prompt = self.build_prompt(text, dl_result, similar_cases, event_info)

        # 重试机制
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个专业的谣言分析助手。"
                                "请基于给定的分析信息，用中文输出清晰、有理有据的解释。"
                                "严格控制在200字以内，直接输出解释内容，不要添加额外格式。"
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,      # 低温度保证解释一致性
                    max_tokens=512,
                    top_p=0.9
                )

                # 更新统计
                self.total_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                explanation = response.choices[0].message.content
                
                # 清理可能的冗余前缀
                explanation = explanation.strip()
                
                return explanation
                
            except Exception as e:
                last_error = e
                print(f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue
        
        # 所有重试都失败
        error_msg = f"解释生成失败: {last_error}"
        print(error_msg)
        return error_msg
    
    def explain_batch(
        self,
        items: List[Dict],
        delay: float = 0.1
    ) -> List[str]:
        """
        批量生成解释
        
        Args:
            items: 包含 text, dl_result, similar_cases, event_info 的字典列表
            delay: 每个请求之间的延迟（秒），避免速率限制
            
        Returns:
            List[str]: 解释文本列表
        """
        explanations = []
        for i, item in enumerate(items):
            if i > 0 and delay > 0:
                time.sleep(delay)
            
            explanation = self.explain(
                text=item['text'],
                dl_result=item['dl_result'],
                similar_cases=item.get('similar_cases', []),
                event_info=item.get('event_info', '')
            )
            explanations.append(explanation)
            
            # 进度提示
            if (i + 1) % 10 == 0:
                print(f"已生成 {i + 1}/{len(items)} 条解释")
        
        return explanations
    
    def get_stats(self) -> Dict:
        """获取调用统计信息"""
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'model': self.model
        }


# 便捷函数：供外部调用
def get_explainer() -> LLMExplainer:
    """获取解释器实例（单例模式）"""
    return LLMExplainer()


def quick_explain(
    text: str,
    dl_result: Dict,
    similar_cases: Optional[List[Dict]] = None,
    event_info: str = ""
) -> str:
    """
    快速生成解释的便捷函数
    
    Args:
        text: 原始推文文本
        dl_result: DL 分类器结果
        similar_cases: 相似案例列表
        event_info: 事件背景信息
        
    Returns:
        str: 解释文本
    """
    explainer = get_explainer()
    return explainer.explain(text, dl_result, similar_cases or [], event_info)


# 测试代码（需要配置 API Key）
if __name__ == '__main__':
    # 测试解释生成
    # 注意：需要先配置 .env 文件中的 SJTU_API_KEY
    
    try:
        explainer = LLMExplainer(
            api_key='sk-UGcjlWJ4hvMHDXNxoVGC5g'
        )
        
        # 模拟 DL 分类器结果
        test_dl_result = {
            'label': 1,
            'confidence': 0.87,
            'keywords': [
                ('witness', 0.23),
                ('says', 0.16),
                ('fired', 0.11)
            ]
        }
        
        # 模拟相似案例
        test_cases = [
            {
                'text': 'Eyewitness claims police used excessive force',
                'label': 1,
                'event': 5,
                'similarity': 0.89
            },
            {
                'text': 'Official statement confirms police followed protocol',
                'label': 0,
                'event': 5,
                'similarity': 0.78
            }
        ]
        
        explanation = explainer.explain(
            text="Witness says police fired tear gas at protesters",
            dl_result=test_dl_result,
            similar_cases=test_cases,
            event_info="关于 Ferguson 事件中警察执法和 Mike Brown 枪击案的讨论"
        )
        
        print("生成的解释:")
        print("-" * 50)
        print(explanation)
        print("-" * 50)
        print(f"统计信息: {explainer.get_stats()}")
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请在 .env 文件中设置 SJTU_API_KEY")