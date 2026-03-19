# RAG Demo

基于**检索增强生成（RAG）**的问答演示项目：使用阿里云 DashScope 的文本嵌入与通义千问大模型，对本地文档做向量检索后再生成回答。支持单轮问答与多轮带上下文对话。

---

## 功能特性

- **文本分块**：从 `data.md` 按段落与标题分块，便于向量化与检索。
- **向量检索**：两种实现可选：
  - **ChromaDB**（`embedding.py`）：持久化向量库，支持多轮对话上下文。
  - **FAISS**（`embedding_faiss.py`）：本地索引，交互式问答循环。
- **多轮对话**：在 ChromaDB 版本中通过维护 `messages` 历史，让模型具备上下文记忆（如追问“他后来怎么样了？”）。

---

## 项目结构

```
RAG_demo/
├── README.md           # 本说明文档
├── .env                # 环境变量（API Key），勿提交
├── data.md             # 原始文档，作为 RAG 知识库来源
├── chunk.py            # 文档读取与分块逻辑
├── embedding.py        # ChromaDB + DashScope，支持多轮对话
├── embedding_faiss.py  # FAISS + DashScope，交互式单轮/多轮
├── chroma_db/          # ChromaDB 持久化目录（运行 embedding.py 后生成）
├── faiss_index.faiss   # FAISS 索引文件（运行 embedding_faiss 后生成）
└── faiss_metadata.json # FAISS 文档 id 映射（同上）
```

---

## 环境与依赖

- **Python**：建议 3.10+
- **主要依赖**：
  - `python-dotenv`：加载 `.env` 配置
  - `dashscope`：阿里云百炼 / 通义千问 API
  - `chromadb`：向量数据库（用于 `embedding.py`）
  - `faiss-cpu` 或 `faiss-gpu`：向量检索（用于 `embedding_faiss.py`）
  - `numpy`：FAISS 所需

安装示例（按需二选一或都装）：

```bash
# ChromaDB 版本
pip install python-dotenv dashscope chromadb

# FAISS 版本
pip install python-dotenv dashscope faiss-cpu numpy
```

---

## 配置

在项目根目录创建 `.env` 文件，填入 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

API Key 可在 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 获取。

---

## 使用说明

### 1. 数据与分块

- 将待检索的文档放入 **`data.md`**（Markdown，按 `\n\n` 分段落，支持 `#` 标题与段落组合成块）。
- 分块逻辑在 **`chunk.py`**，可直接运行查看分块结果：

```bash
python chunk.py
```

### 2. ChromaDB 版本（`embedding.py`）

- **首次使用**：在 `embedding.py` 的 `if __name__ == "__main__"` 中取消注释 `create_db()`，运行一次以构建向量库；之后可注释掉避免重复建库。
- **单轮 RAG**：使用“方式一”的注释代码：`query_db(question)` + 拼 prompt 调用 `Generation.call`。
- **多轮对话（带上下文）**：使用“方式二”，维护 `history`，每轮调用 `chat_with_context(question, history)`，将返回的 `(reply, new_history)` 中的 `new_history` 作为下一轮的 `history` 传入。

直接运行示例（默认执行多轮示例）：

```bash
python embedding.py
```

### 3. FAISS 版本（`embedding_faiss.py`）

- 运行后会先执行 `create_db()` 构建索引并保存到 `faiss_index.faiss` 与 `faiss_metadata.json`；再次运行会 `load_db()` 加载已有索引。
- 进入交互式问答：输入问题回车得到回答，输入 `q` 退出。

```bash
python embedding_faiss.py
```

---

## 模型与参数

| 用途     | 模型 / 配置           | 说明                    |
|----------|------------------------|-------------------------|
| 文本嵌入 | `text-embedding-v1`    | DashScope 嵌入，维度见各文件 |
| 生成回答 | `qwen-plus`            | 通义千问                |
| 向量维度 | embedding.py: 1024；embedding_faiss.py: 1536 | 与 DashScope 调用一致 |

如需更换模型或维度，请修改各文件顶部的 `EMBEDDING_MODEL` / `LLM_MODEL` 及 `dimension` 等常量。

---

## 注意事项

- 向量库（ChromaDB 目录或 FAISS 索引）需在**修改 `data.md` 或分块逻辑后**重新构建（ChromaDB 可重新执行 `create_db()`，FAISS 重新运行一次 `embedding_faiss.py` 的建库流程）。
- 多轮对话会随轮数增加 token 消耗，长对话可考虑对 `history` 做截断或摘要后再传入。
- `.env` 和 API Key 请勿提交到版本库。

---

## 许可与参考

- 本项目为 RAG 学习与演示用途；文档内容（如 `data.md`）请自行确保版权与使用合规。
- DashScope 使用方式参见 [阿里云百炼 API 文档](https://www.alibabacloud.com/help/zh/model-studio/qwen-api-via-dashscope)。
