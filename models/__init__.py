"""
models 模块 — 姜新晨 实现

导出（遵循 DEVELOPMENT.md 接口契约）:
    RumorClassifier  — BERT 分类模型类
    load_model       — 加载模型 + tokenizer
    predict          — 分类推理 + 关键词提取 (text, event_id) -> dict
"""

from .classifier import RumorClassifier, load_model
from .keyword_extractor import predict
