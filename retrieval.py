# retrieval.py
# 相似案例检索模块 - 使用 sentence-transformers 实现
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple


class CaseRetriever:
    """
    案例检索器：使用 sentence-transformers 将训练集推文编码为向量，
    推理时计算余弦相似度，返回最相似的 top_k 条推文。
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化检索器
        
        Args:
            model_name: sentence-transformers 模型名称
                       all-MiniLM-L6-v2 轻量快速，适合本任务
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.texts: Optional[List[str]] = None
        self.labels: Optional[List[int]] = None
        self.events: Optional[List[int]] = None
        self.index_path = 'data/index.pkl'
        
    def build_index(self, csv_path: str, force_rebuild: bool = False) -> None:
        """
        构建索引：编码训练集全部推文并保存到本地
        
        Args:
            csv_path: 训练集 CSV 文件路径
            force_rebuild: 是否强制重建索引（忽略已有索引）
        """
        # 检查是否已有索引文件
        if not force_rebuild and os.path.exists(self.index_path):
            print(f"索引文件已存在: {self.index_path}")
            self.load_index()
            return
        
        print(f"正在读取数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 对文本进行简单清洗（去除 URL，保留可读性）
        self.texts = df['text'].apply(self._clean_text).tolist()
        self.labels = df['label'].tolist()
        self.events = df['event'].tolist()
        
        print(f"正在编码 {len(self.texts)} 条推文...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )
        
        print(f"编码完成，向量维度: {self.embeddings.shape}")
        self._save_index()
        
    def _clean_text(self, text: str) -> str:
        """简单清洗文本，保持可读性"""
        import re
        # 去除 URL
        text = re.sub(r'http\S+', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _save_index(self) -> None:
        """保存索引到本地"""
        os.makedirs('data', exist_ok=True)
        data = {
            'embeddings': self.embeddings,
            'texts': self.texts,
            'labels': self.labels,
            'events': self.events,
            'model_name': self.model_name
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"索引已保存至: {self.index_path}")
    
    def load_index(self, index_path: Optional[str] = None) -> bool:
        """
        从本地加载索引
        
        Args:
            index_path: 索引文件路径，默认使用 self.index_path
            
        Returns:
            bool: 是否加载成功
        """
        path = index_path or self.index_path
        if not os.path.exists(path):
            print(f"索引文件不存在: {path}")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.texts = data['texts']
        self.labels = data['labels']
        self.events = data['events']
        
        # 检查模型是否一致（可选警告）
        if data.get('model_name') != self.model_name:
            print(f"警告: 索引使用的模型 ({data.get('model_name')}) "
                  f"与当前模型 ({self.model_name}) 不一致")
        
        print(f"索引加载成功，包含 {len(self.texts)} 条推文")
        return True
    
    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        检索与查询文本最相似的 top_k 条训练推文
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 包含 text, label, event, similarity 的字典列表
        """
        if self.embeddings is None:
            raise ValueError("索引未加载，请先调用 build_index() 或 load_index()")
        
        # 清洗查询文本
        cleaned_query = self._clean_text(query_text)
        
        # 编码查询
        query_vec = self.model.encode([cleaned_query], convert_to_numpy=True)
        
        # 计算余弦相似度（已归一化的向量直接点积）
        similarities = np.dot(self.embeddings, query_vec.T).squeeze()
        
        # 如果是一维数组，转换为标量
        if similarities.ndim == 0:
            similarities = np.array([similarities])
        
        # 获取 top_k 索引
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'label': self.labels[idx],
                'event': self.events[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def search_with_diversity(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        检索时强制保证标签多样性：至少包含不同标签的案例
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 包含不同标签的案例列表
        """
        # 先检索 top_k * 2 个候选
        candidates = self.search(query_text, top_k=top_k * 2)
        
        # 按标签分组
        label_groups = {0: [], 1: []}
        for case in candidates:
            label_groups[case['label']].append(case)
        
        results = []
        # 优先保证多样性：从不同标签中各取一些
        # 先取每个标签中相似度最高的
        for label in [0, 1]:
            if label_groups[label]:
                results.append(label_groups[label][0])
                label_groups[label] = label_groups[label][1:]
        
        # 如果还需要更多，按相似度从高到低补充
        remaining = []
        for label in [0, 1]:
            remaining.extend(label_groups[label])
        remaining.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 补充到 top_k 条
        while len(results) < top_k and remaining:
            results.append(remaining.pop(0))
        
        return results[:top_k]


# 便捷函数：供外部调用
def build_retrieval_index(csv_path: str = 'rumer2026/train.csv') -> CaseRetriever:
    """构建检索索引的便捷函数"""
    retriever = CaseRetriever()
    retriever.build_index(csv_path)
    return retriever


def get_retriever() -> CaseRetriever:
    """获取已加载索引的检索器（单例模式）"""
    retriever = CaseRetriever()
    if not retriever.load_index():
        # 如果索引不存在，自动构建
        retriever.build_index('rumer2026/train.csv')
    return retriever


# 测试代码
if __name__ == '__main__':
    # 测试检索功能
    retriever = get_retriever()
    
    test_text = "Police fired tear gas at protesters"
    print(f"查询: {test_text}\n")
    
    results = retriever.search(test_text, top_k=3)
    for i, case in enumerate(results):
        print(f"案例{i+1}:")
        print(f"  文本: {case['text']}")
        print(f"  标签: {'谣言' if case['label'] == 1 else '非谣言'}")
        print(f"  事件: {case['event']}")
        print(f"  相似度: {case['similarity']:.4f}\n")