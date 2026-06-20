"""
分类器模型 — AutoModel 通用版
支持: bert-base-uncased, roberta-base, roberta-large, etc.

对外接口（供 inference.py 调用）:
    load_model(checkpoint_path, device) -> (model, tokenizer)

用法:
    from models.classifier import RumorClassifier, load_model
    model, tokenizer = load_model("checkpoints/best_model.pt")
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RumorClassifier(nn.Module):
    """基于预训练模型的谣言检测分类器

    Architecture（当 use_event_embedding=True 时）:
        Encoder (BERT/RoBERTa/...)
        → [CLS]/<s> token embedding (hidden_size)
        Event ID → Embedding(num_events, event_emb_dim)
        → concat([CLS], event_vec) (hidden_size + event_emb_dim)
        → Linear(..., 256) + ReLU + Dropout
        → Linear(256, num_classes) → logits

    Architecture（当 use_event_embedding=False 时，向后兼容旧 [EVENT_N] token 架构）:
        Encoder → [CLS] → classifier head
    """

    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2,
                 dropout: float = 0.3, num_events: int = 7, event_emb_dim: int = 32,
                 use_event_embedding: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_event_embedding = use_event_embedding
        self.num_events = num_events
        self.event_emb_dim = event_emb_dim
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation='eager')
        self.hidden_size = self.encoder.config.hidden_size

        if use_event_embedding:
            self.event_embedding = nn.Embedding(num_events, event_emb_dim)
            classifier_input_dim = self.hidden_size + event_emb_dim
        else:
            self.event_embedding = None
            classifier_input_dim = self.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, event_ids=None, output_attentions=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] or <s>
        cls_embedding = self.dropout(cls_embedding)

        if self.use_event_embedding and self.event_embedding is not None and event_ids is not None:
            event_vec = self.event_embedding(event_ids)
            combined = torch.cat([cls_embedding, event_vec], dim=-1)
        else:
            combined = cls_embedding

        logits = self.classifier(combined)

        if output_attentions:
            return logits, outputs.attentions
        return logits


def load_model(checkpoint_path: str = "checkpoints/best_model.pt",
               device: str = "cpu"):
    """加载训练好的分类模型和 tokenizer

    Args:
        checkpoint_path: 模型权重文件路径
        device: 推理设备 ("cpu" | "cuda")

    Returns:
        (model, tokenizer)

    Usage:
        from models.classifier import load_model
        model, tokenizer = load_model("checkpoints/best_model.pt")
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model_name = checkpoint.get('model_name', 'bert-base-uncased')
    use_event_embedding = checkpoint.get('use_event_embedding', False)
    num_events = checkpoint.get('num_events', 7)
    event_emb_dim = checkpoint.get('event_emb_dim', 32)

    model = RumorClassifier(
        model_name=model_name,
        num_classes=checkpoint.get('num_classes', 2),
        dropout=checkpoint.get('dropout', 0.3),
        num_events=num_events,
        event_emb_dim=event_emb_dim,
        use_event_embedding=use_event_embedding
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 向后兼容：无论是否启用 event embedding，tokenizer 保留 [EVENT_N] tokens
    # 避免旧 checkpoint 的 token embeddings 因 vocab 大小不一致而错位
    event_tokens = [f"[EVENT_{i}]" for i in range(num_events)]
    tokenizer.add_tokens(event_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # 初始化 keyword_extractor 推理状态
    from models.keyword_extractor import _init_predictor
    _init_predictor(model, tokenizer, device)

    val_acc = checkpoint.get('val_accuracy', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"  [OK] 模型加载成功 (model={model_name}, use_event_embedding={use_event_embedding}, epoch={epoch}, val_acc={val_acc}, device={device})")
    return model, tokenizer


# ============================================================
# 自测
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("分类器模块自测")
    print("=" * 60)

    print("\n[1] Event Embedding 模型初始化 (bert-base-uncased, use_event_embedding=True):")
    model = RumorClassifier(model_name="bert-base-uncased", num_classes=2, dropout=0.3,
                            num_events=7, event_emb_dim=32, use_event_embedding=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  Hidden size: {model.hidden_size}, event_emb_dim: {model.event_emb_dim}")

    print("\n[2] Event Embedding 前向传播测试 (event_ids=Tensor):")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))

    test_texts = [
        "Police fired tear gas at protesters",
        "Gurlitt collection accepted by museum",
    ]
    inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    event_ids = torch.tensor([1, 0])

    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'], event_ids=event_ids)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

    for i, text in enumerate(test_texts):
        print(f"  输入 [event={event_ids[i].item()}]: {text[:60]}...")
        print(f"    probs: {probs[i].tolist()}, pred: {preds[i].item()}")
    print("  [OK] Event Embedding 前向传播正常")

    print("\n[3] 兼容模式 (use_event_embedding=False):")
    model_plain = RumorClassifier(model_name="bert-base-uncased", num_classes=2,
                                  dropout=0.3, use_event_embedding=False)
    model_plain.eval()
    with torch.no_grad():
        logits = model_plain(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=-1)
    print(f"  输出形状: {logits.shape}")
    print(f"  [OK] 纯文本模式正常")

    print(f"\n{'='*60}")
    print("自测完成")
