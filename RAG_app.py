import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()

import glob
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ==========================================
# 配置
# ==========================================
DB_PATH = "./chroma_db"
COLLECTION_NAME = "courses"
KNOWLEDGE_DIR = "knowledge_base"

SYSTEM_PROMPT = """你是一个专业的课程助手。请严格根据下面提供的【参考资料】来回答用户的问题。
如果参考资料里没有相关信息，请直接回答"根据现有资料，我无法回答这个问题"，绝对不要自己编造。

【参考资料】：
{context}"""


# ==========================================
# 用 @st.cache_resource 缓存，模型/向量库只加载一次
# ==========================================
@st.cache_resource
def get_collection():
    """加载向量库。首次运行会建库，之后缓存复用（不再重算向量）。"""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len
        )
        documents, metadatas, ids = [], [], []
        for file_path in glob.glob(f"{KNOWLEDGE_DIR}/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            file_name = os.path.basename(file_path)
            for i, chunk in enumerate(text_splitter.split_text(content)):
                documents.append(chunk)
                metadatas.append({"file": file_name})
                ids.append(f"{file_name}-{i}")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection


@st.cache_resource
def get_client():
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


collection = get_collection()
client = get_client()


# ==========================================
# 页面 UI
# ==========================================
st.set_page_config(page_title="课程知识库助手", page_icon="📚")
st.title("📚 课程知识库问答助手")
st.caption(f"已加载 {collection.count()} 个知识块，基于你的课程集锦文件夹")

# 聊天历史（存到 session_state，页面刷新也不丢）
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题，例如：微软人工智能公开课讲了什么？"):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 检索
    results = collection.query(query_texts=[prompt], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join([f"【文件名：{m['file']}】\n{d}" for d, m in zip(docs, metas)])

    # 3. 生成 + 展示
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
        st.markdown(answer)

        # 展示命中的来源文件，方便判断检索质量
        with st.expander("查看命中的来源"):
            for m in metas:
                st.write(f"- {m['file']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
