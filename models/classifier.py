"""
BERT 分类器模型 — 姜新晨 实现

模型结构: BERT-base-uncased + 2 层 MLP 分类头

对外接口（供 inference.py 调用）:
    load_model(checkpoint_path) -> (model, tokenizer)

用法:
    from models.classifier import RumorClassifier, load_model

    model, tokenizer = load_model("checkpoints/best_model.pt")
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class RumorClassifier(nn.Module):
    """
    基于 BERT 的谣言检测分类器

    Architecture:
        BERT-base-uncased (12 layers, 768 hidden)
        → [CLS] token embedding (768-dim)
        → Linear(768, 256) + ReLU + Dropout(0.3)
        → Linear(256, 2)  → logits

    Args:
        num_classes: 分类数 (默认 2: 非谣言/谣言)
        dropout:     Dropout 概率 (默认 0.3)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        # 使用 eager attention 以支持 output_attentions（关键词提取需要）
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            attn_implementation='eager'
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False
    ):
        """
        Args:
            input_ids:        (batch_size, seq_len)
            attention_mask:   (batch_size, seq_len)
            output_attentions: 是否返回 attention weights（关键词提取用）

        Returns:
            若 output_attentions=False: logits (batch_size, num_classes)
            若 output_attentions=True:  (logits, attentions)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # [CLS] token 的 hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)              # (batch, num_classes)

        if output_attentions:
            return logits, outputs.attentions
        return logits


# ============================================================
# 模型加载接口（供 inference.py 调用）
# ============================================================

def load_model(
    checkpoint_path: str = "checkpoints/best_model.pt",
    device: str = "cpu"
) -> tuple[RumorClassifier, BertTokenizer]:
    """
    加载训练好的分类模型和 tokenizer

    严格按照 DEVELOPMENT.md 接口契约:
        def load_model(checkpoint_path: str = "checkpoints/best_model.pt")

    加载后会自动初始化 keyword_extractor 模块的 predict() 函数，
    使得 predict(text, event_id) 可以直接调用。

    Args:
        checkpoint_path: 模型权重文件路径
        device:          推理设备 ("cpu" | "cuda")

    Returns:
        (model, tokenizer)

    Usage:
        from models.classifier import load_model
        from models.keyword_extractor import predict

        model, tokenizer = load_model("checkpoints/best_model.pt")
        result = predict("Police fired tear gas at protesters", event_id=1)
    """
    # 检测可用设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 创建模型
    model = RumorClassifier(
        num_classes=checkpoint.get('num_classes', 2),
        dropout=checkpoint.get('dropout', 0.3)
    )

    # 添加 event tokens（与训练时一致）
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)

    # 扩展 embedding 层以容纳新 token
    model.bert.resize_token_embeddings(len(tokenizer))

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 初始化 keyword_extractor 模块的推理状态
    # 这样 predict(text, event_id) 就能直接使用
    from models.keyword_extractor import _init_predictor
    _init_predictor(model, tokenizer, device)

    # 打印信息
    val_acc = checkpoint.get('val_accuracy', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"  [OK] 模型加载成功 (epoch={epoch}, val_acc={val_acc}, device={device})")

    return model, tokenizer


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BERT 分类器模块自测")
    print("=" * 60)

    # 1. 模型初始化
    print("\n[1] 模型初始化:")
    model = RumorClassifier(num_classes=2, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  [OK] 模型创建成功")

    # 2. 前向传播测试
    print("\n[2] 前向传播测试:")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    event_tokens = [f"[EVENT_{i}]" for i in range(7)]
    tokenizer.add_tokens(event_tokens)
    model.bert.resize_token_embeddings(len(tokenizer))

    test_texts = [
        "[EVENT_1] Police fired tear gas at protesters",
        "[EVENT_0] Gurlitt collection accepted by museum",
    ]
    inputs = tokenizer(
        test_texts, padding=True, truncation=True,
        max_length=64, return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

    for i, text in enumerate(test_texts):
        print(f"  输入: {text[:60]}...")
        print(f"    logits: {logits[i].tolist()}")
        print(f"    prob: {probs[i].tolist()}")
        print(f"    pred: {preds[i].item()}")
    print(f"  [OK] 前向传播正常")

    # 3. Attention 输出测试
    print("\n[3] Attention 输出测试:")
    with torch.no_grad():
        logits, attentions = model(
            inputs['input_ids'], inputs['attention_mask'],
            output_attentions=True
        )
    print(f"  attention 层数: {len(attentions)}")
    print(f"  最后一层 attention shape: {attentions[-1].shape}")
    print(f"  [OK] Attention 输出正常")

    print("\n" + "=" * 60)
    print("自测完成")
