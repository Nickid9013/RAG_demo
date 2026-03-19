"""
RAG 演示：文本嵌入与向量检索模块。
使用 DashScope 文本嵌入 API 和 ChromaDB 实现文档向量化与相似度检索。
"""

import dotenv
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from http import HTTPStatus
import chunk
import chromadb
import os

# 从 .env 加载环境变量（如 DASHSCOPE_API_KEY）
load_dotenv()

# 配置 DashScope API 密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

# 模型配置：嵌入模型用于向量化，LLM 模型用于生成回答
EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-plus"

# 持久化 ChromaDB 客户端与集合（用于存储文档块及其向量）
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
chromadb_collection = chromadb_client.get_or_create_collection(name="linhuchong")


def embed(text: str, store: bool = True) -> list[float]:
    """
    调用 DashScope 将文本转为 1024 维向量。

    :param text: 待嵌入的文本
    :param store: True 表示文档（入库用），False 表示查询（检索用），影响 text_type
    :return: 嵌入向量
    :raises RuntimeError: API 调用失败时抛出
    """
    text_type = "document" if store else "query"
    resp = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=text,
        text_type=text_type,
        dimension=1024
    )

    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope API 调用失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def create_db() -> None:
    """
    根据 chunk 模块提供的文本块构建向量库。 
    遍历所有块，生成嵌入并写入 ChromaDB 集合。
    """
    for idx, c in enumerate(chunk.get_chunks()):
        print(f"正在处理：{c}")
        embedding = embed(c, store=True)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding,
        )
    print("数据库创建完成")


def query_db(query: str) -> list[str]:
    """
    根据用户问题在向量库中检索最相关的文档块。

    :param query: 用户问题文本
    :return: 与问题最相关的若干文档块（默认取前 3 条）
    """
    question_embedding = embed(query, store=False)
    results = chromadb_collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    return results["documents"][0]


def chat_with_context(
    question: str,
    history: list[dict],
    system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    """
    带上下文的多轮对话：在 RAG 检索基础上，把历史对话一并传给模型，使模型具备“记住之前说过什么”的能力。

    实现要点：
    - 通义千问 API 本身无状态，不保存历史，因此每次请求都要传入完整的 messages 列表。
    - 使用 messages 格式（role: system/user/assistant）并设置 result_format='message'。
    - 每轮：把当前问题做 RAG 检索 → 用「系统提示 + 历史消息 + 本轮(问题+检索信息)」调用 API → 把助手回复追加到 history 再返回。

    :param question: 用户本轮问题
    :param history: 之前的对话列表，每项为 {"role": "user"|"assistant", "content": "..."}
    :param system_prompt: 可选系统提示，用于设定角色或 RAG 说明；若为 None 则使用默认说明
    :return: (本轮模型回复, 更新后的 history，可直接作为下一轮的 history 传入)
    """
    default_system = (
        "你是一个基于给定资料回答问题的助手。请根据「信息」中的内容回答用户问题；"
        "若信息中没有答案，请回答“我不知道”。可以结合对话历史理解用户的追问。"
    )
    system_content = system_prompt if system_prompt is not None else default_system

    # 当前问题的 RAG 检索
    chunks = query_db(question)
    info = "\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    user_content = f"信息：\n{info}\n\n用户问题：{question}"

    # 组装完整 messages：系统 + 历史 + 本轮用户消息
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    response = dashscope.Generation.call(
        model=LLM_MODEL,
        messages=messages,
        result_format="message",
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Generation 调用失败: {response.message}")

    raw = response.output.choices[0].message.content
    # 兼容：纯文本时多为 str，多模态时为 list[dict]，取首项 text
    reply = raw[0]["text"] if isinstance(raw, list) and raw and isinstance(raw[0], dict) else raw
    # 更新历史：追加本轮 user 与 assistant，供下一轮使用
    new_history = history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": reply},
    ]
    return reply, new_history


if __name__ == "__main__":
    # 首次运行可先执行以构建向量库
    # create_db()

    # ---------- 方式一：单轮 RAG（无上下文记忆）----------
    # question = "令狐冲的身上发生了什么事？"
    # chunks = query_db(question)
    # prompt = f"问题：{question}\n信息：{chunks}\n请根据信息回答，若无答案请回答“我不知道”。"
    # response = dashscope.Generation.call(model=LLM_MODEL, prompt=prompt)
    # print(response.output.text)

    # ---------- 方式二：多轮对话（带上下文）----------
    # 维护 history，每轮调用 chat_with_context 并传入上一轮返回的 history
    # history: list[dict] = []
    # q1 = "令狐冲的身上发生了什么事？"
    # reply1, history = chat_with_context(q1, history)
    # print(f"用户: {q1}\n助手: {reply1}\n")

    # q2 = "他后来怎么样了？"  # 模型能结合上一轮回答理解“他”指令狐冲
    # reply2, history = chat_with_context(q2, history)
    # print(f"用户: {q2}\n助手: {reply2}\n")

    # ---------方式三：终端循环对话----------
    history: list[dict] = []
    print("欢迎使用令狐冲助手！输入 'q' 退出。")
    while True:
        question = input("用户: ")
        if question.lower() == "q":
            break
        reply, history = chat_with_context(question, history)
        print(f"助手: {reply}")