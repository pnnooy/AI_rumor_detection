"""
关键词提取（基于 BERT Attention Weights）— 姜新晨 实现

方法: 取 BERT 最后一层所有 head 的 [CLS] token 对其他 token 的注意力分数,
      平均后提取 top-k 个 token 作为关键词。

对外接口（严格按照 DEVELOPMENT.md 契约）:
    predict(text, event_id) -> dict     # 分类推理 + 关键词提取
    _init_predictor(model, tokenizer, device)  # 初始化推理状态

使用流程:
    from models.classifier import load_model
    from models.keyword_extractor import predict

    model, tokenizer = load_model("checkpoints/best_model.pt")
    # load_model 内部会调用 _init_predictor() 完成初始化
    result = predict("Police fired tear gas at protesters", event_id=1)
    # => {"label": 1, "confidence": 0.87, "keywords": [("police", 0.23), ...]}
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer


# ============================================================
# 模块级状态（供 predict() 使用）
# ============================================================

_model = None
_tokenizer = None
_device = "cpu"


def _init_predictor(model, tokenizer: BertTokenizer, device: str = "cpu"):
    """
    初始化推理状态 — 由 load_model() 内部调用

    Args:
        model:     已加载的 RumorClassifier 模型
        tokenizer: BERT tokenizer（已添加 event tokens）
        device:    推理设备
    """
    global _model, _tokenizer, _device
    _model = model
    _tokenizer = tokenizer
    _device = device
    _model.eval()


# ============================================================
# 关键词提取（内部函数）
# ============================================================

def _extract_keywords(
    model,
    tokenizer: BertTokenizer,
    text: str,
    event_id: int,
    top_k: int = 5,
) -> list:
    """
    从 BERT attention weights 中提取模型最关注的 top_k 个词

    方法:
        1. 拼接 event token，tokenize
        2. 前向传播，获取 attention weights
        3. 取最后一层，所有 head 平均
        4. 提取 [CLS] token 对其他 token 的注意力分数
        5. 排除特殊 token ([CLS], [SEP], [PAD], [EVENT_X])
        6. 将 subword token 合并为完整词
        7. 按分数排序，返回 top_k
    """
    from preprocess import clean_text

    # 清洗 + event token
    cleaned = clean_text(text)
    text_with_event = f"[EVENT_{event_id}] {cleaned}"

    # Tokenize
    inputs = tokenizer(
        text_with_event,
        return_tensors='pt',
        truncation=True,
        max_length=64
    )
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    # 前向传播
    with torch.no_grad():
        _, attentions = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            output_attentions=True
        )

    # 最后一层, 所有 head 平均, [CLS] 对其他 token 的注意力
    last_layer_attn = attentions[-1]               # (1, num_heads, seq_len, seq_len)
    cls_attn = last_layer_attn[:, :, 0, :]          # (1, num_heads, seq_len)
    cls_attn = cls_attn.mean(dim=1)                  # (1, seq_len)

    # token → attention score
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    SPECIAL_TOKENS = {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}
    EVENT_PREFIX = '[EVENT_'

    token_scores = []
    for i, (token, score) in enumerate(zip(tokens, cls_attn[0].cpu())):
        if token in SPECIAL_TOKENS or token.startswith(EVENT_PREFIX):
            continue
        token_scores.append((token, score.item()))

    # 合并 subword token
    merged = []
    for token, score in token_scores:
        if token.startswith('##') and merged:
            merged[-1] = (merged[-1][0] + token[2:], max(merged[-1][1], score))
        else:
            merged.append((token, score))

    # 排序 + 归一化
    merged.sort(key=lambda x: x[1], reverse=True)
    top_tokens = merged[:top_k]
    if top_tokens:
        total = sum(s for _, s in top_tokens)
        if total > 0:
            top_tokens = [(w, s / total) for w, s in top_tokens]

    return top_tokens


# ============================================================
# 对外接口 — predict(text, event_id) → dict
# ============================================================

def predict(text: str, event_id: int) -> dict:
    """
    对单条推文做分类预测 + 关键词提取

    严格按照 DEVELOPMENT.md 接口契约:
        def predict(text: str, event_id: int) -> dict

    Args:
        text:     原推文文本（未经清洗）
        event_id: 事件 ID (0-6)

    Returns:
        {
            "label":      int,         # 0=非谣言, 1=谣言
            "confidence": float,       # 0.0 ~ 1.0
            "keywords":   [            # top-5 关键词，按 attention 分数降序
                ("police", 0.23),
                ("witness", 0.16),
                ...
            ]
        }

    Raises:
        RuntimeError: 如果模型尚未初始化（需先调用 load_model()）

    Example:
        from models.classifier import load_model
        from models.keyword_extractor import predict

        model, tokenizer = load_model("checkpoints/best_model.pt")
        result = predict("Police fired tear gas at protesters", event_id=1)
        print(result)
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "模型尚未初始化。请先调用 load_model() 加载模型。\n"
            "  from models.classifier import load_model\n"
            "  model, tokenizer = load_model('checkpoints/best_model.pt')"
        )

    from preprocess import clean_text

    cleaned = clean_text(text)
    text_with_event = f"[EVENT_{event_id}] {cleaned}"

    inputs = _tokenizer(
        text_with_event,
        return_tensors='pt',
        truncation=True,
        max_length=64
    )
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(
            inputs['input_ids'],
            inputs['attention_mask'],
            output_attentions=False
        )

    probs = F.softmax(logits, dim=-1)
    pred_label = int(torch.argmax(probs, dim=-1).item())
    confidence = float(probs[0, pred_label].item())

    keywords = _extract_keywords(_model, _tokenizer, text, event_id)

    return {
        "label": pred_label,
        "confidence": confidence,
        "keywords": keywords
    }


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("关键词提取模块自测")
    print("=" * 60)

    from transformers import BertTokenizer
    from models.classifier import RumorClassifier
    from preprocess import clean_text

    # 初始化
    print("\n[1] 初始化模型和 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)

    model = RumorClassifier(num_classes=2, dropout=0.3)
    model.bert.resize_token_embeddings(len(tokenizer))
    _init_predictor(model, tokenizer, device="cpu")
    print("  [OK] 初始化完成")

    # 测试 predict（两参数版本，符合契约）
    print("\n[2] predict(text, event_id) 测试:")
    test_texts = [
        ("Police fired tear gas at protesters in Ferguson", 1),
        ("Gurlitt collection accepted by Swiss museum", 0),
    ]
    for text, event_id in test_texts:
        result = predict(text, event_id)
        print(f"  文本: {text[:55]}...")
        print(f"    label: {result['label']}, confidence: {result['confidence']:.4f}")
        print(f"    keywords: {result['keywords']}")
        print()

    # 验证返回格式
    print("[3] 返回格式验证:")
    result = predict("Test tweet about Ferguson", 1)
    assert isinstance(result['label'], int), "label 必须是 int"
    assert isinstance(result['confidence'], float), "confidence 必须是 float"
    assert 0.0 <= result['confidence'] <= 1.0, "confidence 必须在 [0,1]"
    assert isinstance(result['keywords'], list), "keywords 必须是 list"
    assert len(result['keywords']) <= 5, "keywords 数量不超过 5"
    for kw in result['keywords']:
        assert isinstance(kw, tuple) and len(kw) == 2, "每个 keyword 是 (str, float)"
        assert isinstance(kw[0], str), "keyword[0] 是 str"
        assert isinstance(kw[1], float), "keyword[1] 是 float"
    print("  [OK] 返回格式符合契约")

    print("\n" + "=" * 60)
    print("自测完成")
    print("=" * 60)
    print("\n[WARN] 未训练的模型（随机权重）关键词无实际意义。")
