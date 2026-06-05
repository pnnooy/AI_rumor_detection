# 开发协作指南

> 三人协作开发流程、Git 分支规范、接口契约、集成 check list

---

## 一、角色与职责

| 成员 | 职责 | 交付物 |
|------|------|--------|
| **姜新晨** | 数据预处理 + BERT 分类器 + 关键词提取 | `models/classifier.py`, `models/keyword_extractor.py`, `preprocess.py`, `train.py`, `checkpoints/best_model.pt` |
| **靳卓达** | 相似案例检索 + LLM 解释生成 | `retrieval.py`, `llm_explainer.py`, `event_context.py`, `data/index.pkl` |
| **韩宇飞** | 项目管理 + 系统集成 + 评估 + 报告 | `inference.py`, `evaluate.py`, `results/`, `figures/`, `report/` |

---

## 二、Git 分支规范

### 分支策略

```
main ─────────────────────────────────────────────────→
  │
  ├── jiang-xinchen/classifier ──→ PR → merge
  ├── jin-zhuoda/explainer  ──→ PR → merge
  └── han-yufei/integration──→ PR → merge
```

### 命名规则

| 分支 | 命名格式 | 示例 |
|------|---------|------|
| 功能分支 | `member-{角色}/{模块名}` | `jiang-xinchen/classifier`, `jin-zhuoda/retrieval` |
| 修复分支 | `fix/{描述}` | `fix/api-timeout`, `fix/preprocess-url` |
| 实验分支 | `exp/{描述}` | `exp/data-augmentation`, `exp/leave-one-event` |

### 工作流

**第一步：每个人从 main 拉自己的分支**
```bash
git checkout main
git pull origin main
git checkout -b jiang-xinchen/classifier   # 姜新晨
git checkout -b jin-zhuoda/explainer    # 靳卓达
git checkout -b han-yufei/integration  # 韩宇飞
```

**第二步：日常开发**
```bash
# 开发中，随时提交到自己的分支
git add <files>
git commit -m "feat: 完成数据预处理函数"
git push origin jiang-xinchen/classifier   # 推到自己的远程分支
```

**第三步：合并到 main**

当模块完成并通过自测后，发起 Pull Request：

1. 到 GitHub 仓库页面 → Pull requests → New pull request
2. base 选 `main`，compare 选你的分支
3. 写清楚 PR 描述：
   - 这个 PR 做了什么
   - 如何测试
   - 新增了哪些文件/依赖
4. 指定 **韩宇飞** 为 reviewer
5. 韩宇飞 review 通过后 → Merge

### Commit 规范

```
feat: 新功能     feat: 添加 attention 关键词提取函数
fix: 修复 bug    fix: 修复 URL 清洗正则表达式遗漏 case
docs: 文档更新   docs: 更新 README API 配置说明
refactor: 重构   refactor: 简化分类器前向传播逻辑
test: 测试相关   test: 添加跨事件评估测试
```

### 注意事项

- ⚠️ **永远不要在 main 分支上直接开发**
- ⚠️ **不要 force push 到 main 分支**
- ⚠️ **merge 前确保代码能在自己机器上跑通**
- ⚠️ **不要提交 `.env` 文件**（已在 `.gitignore` 中）
- ⚠️ **模型权重文件不要提交**（`checkpoints/` 在 `.gitignore` 中，团队成员私下传输）

---

## 三、接口契约

> ⚠️ 姜新晨 和 靳卓达 必须严格按照以下接口实现，韩宇飞 的集成代码依赖这些接口。

### 姜新晨 的接口

```python
def predict(text: str, event_id: int) -> dict:
    """
    输入: 推文文本 + 事件ID
    输出: {
        "label": int,        # 0 或 1
        "confidence": float, # 0.0 ~ 1.0
        "keywords": [        # top-5 关键词，按 attention 分数降序
            ("police", 0.23),
            ("witness", 0.16),
            ...
        ]
    }
    """

def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """加载模型和 tokenizer，供 inference.py 调用"""
```

### 靳卓达 的接口

```python
class CaseRetriever:
    def build_index(self, csv_path: str):
        """构建训练集向量索引"""
    
    def load_index(self, index_path: str = "data/index.pkl"):
        """从文件加载索引"""
    
    def search(self, query_text: str, top_k: int = 3) -> list:
        """
        Returns: [
            {"text": str, "label": int, "event": int, "similarity": float},
            ...
        ]
        """

def generate_explanation(text: str, dl_result: dict, cases: list, event_context: str) -> str:
    """
    输入: 原推文 + DL 结果 + 相似案例 + 事件背景
    输出: 中文解释字符串 (150-300字)
    """
```

### 数据格式约定

| 约定 | 说明 |
|------|------|
| 文本编码 | UTF-8 |
| CSV 读取 | 使用 `pd.read_csv()` 默认参数 |
| label 含义 | 0 = 非谣言, 1 = 谣言 |
| event_id | 0-6 整数 |
| 路径 | 所有路径使用相对路径，基于项目根目录 |
| 随机种子 | 全局 `seed=42` |

---

## 四、韩宇飞 的工作阶段

### 阶段一：搭骨架（第 1 周，不依赖任何人）

- [ ] 创建项目目录结构（`models/`, `data/`, `checkpoints/`, `logs/`, `results/`, `figures/`, `report/`）
- [ ] 编写 `requirements.txt`
- [ ] 编写 `.env.example`
- [ ] 搭建 `inference.py` 骨架（接口定义好，函数体留 `pass` 或 `raise NotImplementedError`）
- [ ] 搭建 `evaluate.py` 骨架（评估指标函数先写好）
- [ ] 初始化 Git，创建 main 分支，推送到 GitHub
- [ ] 帮 姜新晨 和 靳卓达 创建他们的开发分支
- [ ] 确保 姜新晨 和 靳卓达 都清楚自己的接口契约

### 阶段二：并行开发期（第 1-2 周，轻度依赖 姜新晨 和 靳卓达）

- [ ] **Git 管理**：关注 姜新晨 和 靳卓达 的 PR，做 code review
- [ ] **报告框架**：打开模板 doc，搭好章节结构
- [ ] **评估脚本**：完善 `evaluate.py`，用假数据验证所有指标计算正确
- [ ] **事件背景**：读各 event 的推文样本，补完 `event_context.py`
- [ ] **定期同步**：每 2-3 天问 姜新晨 和 靳卓达 进度，确认接口无变更
- [ ] **准备演示**：确认 `--no-llm` 模式可以先跑通分类部分

### 阶段三：集成联调（第 3 周，依赖 姜新晨 和 靳卓达 交付）

- [ ] 姜新晨 的 `predict()` 函数在 韩宇飞 的环境跑通
- [ ] 靳卓达 的 `CaseRetriever.search()` + `generate_explanation()` 在 韩宇飞 的环境跑通
- [ ] 端到端推理管道串联成功
- [ ] val.csv 全量（401条）推理通过，结果写入 `results/val_results.csv`
- [ ] `--no-llm` 模式验证可用
- [ ] API 调用正常（校园网/VPN 环境下）
- [ ] 随机抽 10 条结果，人工检查解释质量

### 阶段四：评估报告（第 3-4 周，不依赖别人）

- [ ] 运行 `evaluate.py` 输出全部评估图表
- [ ] 错误案例分析（选 20 个分类错误的）
- [ ] Leave-One-Event-Out 实验（如果时间允许）
- [ ] 撰写报告：
  - 韩宇飞 写：架构概述、检测结果分析、可解释性分析、工作总结
  - 姜新晨 补充：数据预处理、分类器训练、关键词提取章节
  - 靳卓达 补充：案例检索、LLM 提示词设计、解释生成章节
- [ ] 最终校对、排版、提交

---

## 五、集成测试 Check List

部署到新环境时，按以下顺序验证：

```bash
# 1. 环境检查
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import sentence_transformers; print('OK')"
python -c "from openai import OpenAI; print('OK')"

# 2. 分类器测试（不调 API）
python inference.py --input rumer2026/val.csv --output results/test.csv --no-llm
# 预期：无报错，输出 CSV 包含 pred_label 和 confidence 列

# 3. 检索测试
python -c "from retrieval import CaseRetriever; r = CaseRetriever(); r.load_index('data/index.pkl'); print(r.search('test query'))"
# 预期：返回 3 条相似案例

# 4. LLM 测试（需要 API key 和校园网/VPN）
python -c "from llm_explainer import LLMExplainer; e = LLMExplainer(); print(e.explain('test', {'label':0,'confidence':0.9,'keywords':[('test',0.5)]}, [], ''))"
# 预期：返回中文解释字符串

# 5. 全量推理
python inference.py --input rumer2026/val.csv --output results/val_results.csv
# 预期：401 条全部处理完，有 label、confidence、keywords、explanation

# 6. 评估
python evaluate.py --input results/val_results.csv --output-dir figures/
# 预期：输出评估指标 + 5 张图
```

---

## 六、常见问题处理

### 分类器加载失败
```
错误: RuntimeError: Error(s) in loading state_dict
原因: 模型定义和保存时不一致
解决: 确认 models/classifier.py 未被修改，通知 姜新晨 重新训练
```

### 姜新晨PI 调用返回空
```
错误: LLM 返回空内容
原因: 请求中缺少 user 角色的消息
解决: 检查 prompt 构建，确保 messages 列表中有一条 role='user' 的消息
```

### 检索索引文件缺失
```
错误: FileNotFoundError: data/index.pkl
原因: 靳卓达 还没有构建索引
解决: 让 靳卓达 运行一次 build_index() 生成索引文件，然后手动传递
```

### 模块导入失败
```
错误: ModuleNotFoundError: No module named 'xxx'
原因: requirements.txt 缺少依赖
解决: 让模块负责人补充 requirements.txt，然后 pip install -r requirements.txt
```
