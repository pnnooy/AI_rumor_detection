"""
关键词提取（基于 Attention Weights）— AutoModel 通用版
支持: bert-base-uncased, roberta-base, roberta-large, etc.

对外接口:
    predict(text, event_id) -> dict     # 分类推理 + 关键词提取
    _init_predictor(model, tokenizer, device)  # 初始化推理状态

使用流程:
    from models.classifier import load_model
    from models.keyword_extractor import predict

    model, tokenizer = load_model("checkpoints/best_model.pt")
    result = predict("Police fired tear gas at protesters", event_id=1)
"""

import torch
import torch.nn.functional as F

_model = None
_tokenizer = None
_device = "cpu"


def _init_predictor(model, tokenizer, device: str = "cpu"):
    global _model, _tokenizer, _device
    _model = model
    _tokenizer = tokenizer
    _device = device
    _model.eval()


def _extract_keywords(model, tokenizer, text: str, event_id: int, top_k: int = 5):
    from preprocess import clean_text

    cleaned = clean_text(text)
    text_with_event = f"[EVENT_{event_id}] {cleaned}"

    inputs = tokenizer(text_with_event, return_tensors='pt', truncation=True, max_length=128)
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        _, attentions = model(
            inputs['input_ids'], inputs['attention_mask'], output_attentions=True
        )

    last_layer_attn = attentions[-1]
    cls_attn = last_layer_attn[:, :, 0, :]
    cls_attn = cls_attn.mean(dim=1)

    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 兼容 BERT 和 RoBERTa 的特殊 token
    SPECIAL_TOKENS = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>'}
    EVENT_PREFIX = '[EVENT_'

    token_scores = []
    for i, (token, score) in enumerate(zip(tokens, cls_attn[0].cpu())):
        if token in SPECIAL_TOKENS or token.startswith(EVENT_PREFIX):
            continue
        token_scores.append((token, score.item()))

    # 合并 subword token（兼容 BERT ## 和 RoBERTa Ġ）
    merged = []
    for token, score in token_scores:
        if token.startswith('##') and merged:
            merged[-1] = (merged[-1][0] + token[2:], max(merged[-1][1], score))
        elif token.startswith('Ġ') and merged:
            merged.append((token[1:], score))
        elif token.startswith('Ġ'):
            merged.append((token[1:], score))
        elif merged and not token.startswith('##'):
            merged.append((token, score))
        else:
            merged.append((token, score))

    merged.sort(key=lambda x: x[1], reverse=True)
    top_tokens = merged[:top_k]
    if top_tokens:
        total = sum(s for _, s in top_tokens)
        if total > 0:
            top_tokens = [(w, s / total) for w, s in top_tokens]

    return top_tokens


def predict(text: str, event_id: int) -> dict:
    """对单条推文做分类预测 + 关键词提取

    Args:
        text: 原推文文本（未经清洗）
        event_id: 事件 ID (0-6)

    Returns:
        {"label": int, "confidence": float, "keywords": [(str, float), ...]}
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "模型尚未初始化。请先调用 load_model() 加载模型。"
        )

    from preprocess import clean_text

    cleaned = clean_text(text)
    text_with_event = f"[EVENT_{event_id}] {cleaned}"

    inputs = _tokenizer(text_with_event, return_tensors='pt', truncation=True, max_length=128)
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(inputs['input_ids'], inputs['attention_mask'], output_attentions=False)

    probs = F.softmax(logits, dim=-1)
    pred_label = int(torch.argmax(probs, dim=-1).item())
    confidence = float(probs[0, pred_label].item())

    keywords = _extract_keywords(_model, _tokenizer, text, event_id)
    return {"label": pred_label, "confidence": confidence, "keywords": keywords}
