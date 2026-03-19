import faiss
import numpy as np
import pickle
import json
import os
from dotenv import load_dotenv
import dashscope
from http import HTTPStatus
import chunk

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

Embedding_model = "text-embedding-v1"
llm_model = "qwen-plus"

Dimension = 1536

faiss_index = faiss.IndexFlatL2(Dimension)

id_to_document = {}

INDEX_PATH = './faiss_index.faiss'
METADATA_PATH = './faiss_metadata.json'

def embed(text: str, store: bool = True) -> list[float]:
    """
    """
    text_type = "document" if store else "query"
    resp = dashscope.TextEmbedding.call(
        model=Embedding_model,
        input=text,
        text_type=text_type,
        dimension=Dimension
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError("DashScope API 调用失败: " + resp.message)
    # print(len(resp.output["embeddings"][0]["embedding"]))
    return resp.output["embeddings"][0]["embedding"]

def create_db() -> None:
    global faiss_index, id_to_document

    faiss_index.reset()
    id_to_document.clear()

    for idx, c in enumerate(chunk.get_chunks()):
        print("正在处理：" + c)
        embedding = embed(c, store=True)

        embeddings_array = np.array([embedding], dtype=np.float32)

        current_id = faiss_index.ntotal

        faiss_index.add(embeddings_array)

        id_to_document[current_id] = c

    faiss.write_index(faiss_index, INDEX_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(id_to_document, f)

    print(f"数据库创建完成， 共{faiss_index.ntotal}条数据")

def load_db() -> None:
    global faiss_index, id_to_document

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'r') as f:
            id_to_document = json.load(f)
        print(f"数据库加载完成， 共{faiss_index.ntotal}条数据")
    else:
        print("数据库不存在， 请先创建数据库")
        
        faiss_index = faiss.IndexFlatL2(Dimension)

def query_db(query: str) -> list[str]:
    global faiss_index, id_to_document

    question_embedding = embed(query, store=False)
    question_embedding = np.array([question_embedding], dtype=np.float32)

    distances, indices = faiss_index.search(question_embedding, k=3)

    results = []
    for idx in indices[0]:
        if idx != -1:
            # print(idx)
            results.append(id_to_document[str(idx)])

    return results

def get_response(query: str) -> str:
    results = query_db(query)

    default_system = (
        "你是一个基于给定资料回答问题的助手。请根据「信息」中的内容回答用户问题；"
        "若信息中没有答案，请回答“我不知道”。可以结合对话历史理解用户的追问。"
    )
    context = "\n".join(results) if isinstance(results, list) else str(results)

    prompt = f"信息：\n{context}\n\n用户问题：{query}"

    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_plus,
        messages=[{"role": "system", "content": default_system}, {"role": "user", "content": prompt}],
        result_format="message",
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Generation 调用失败: {response.message}")
    return response.output.choices[0].message.content

if __name__ == "__main__":
    create_db()
    print("数据库创建完成")

    load_db()
    print("请提问：（输入 'q' 退出）")
    while True:
        question = input("用户: ")
        if question.lower() == 'q':
            break
        print("助手: " + get_response(question))
    print("再见！")
