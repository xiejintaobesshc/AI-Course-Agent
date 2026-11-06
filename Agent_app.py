import os

# 强制离线，秒加载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ==========================================
# 1. 页面与资源初始化
# ==========================================
st.set_page_config(page_title="AI 智能课程助手", page_icon="🤖", layout="centered")
st.title("🤖 AI 智能课程助手 (Agent 版)")
st.caption("具备查阅知识库 + 数学计算 + 自主决策能力")


@st.cache_resource
def init_resources():
    # 初始化大模型
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    # 初始化向量库
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="courses", embedding_function=embedding_fn)
    return client, collection


client, collection = init_resources()


# ==========================================
# 2. 定义工具 (在后台真正执行的 Python 代码)
# ==========================================
def search_courses(query):
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return "\n\n".join([f"【文件名：{m['file']}】\n{d}" for d, m in zip(docs, metas)])


def calculate(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# ==========================================
# 3. 声明工具 (告诉大模型有哪些武器)
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "当用户询问课程大纲、课程内容、授课老师等知识库相关问题时，调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "当用户需要进行数学计算时，调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，比如 '85+90'"}},
                "required": ["expression"],
            },
        }
    }
]

# ==========================================
# 4. 聊天界面与逻辑
# ==========================================
# 初始化历史记录 (注意：我们存的是 OpenAI 标准格式的 messages)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个智能课程助手。如果用户问课程相关，请调用 search_courses。如果是算数，请调用 calculate。如果只是闲聊，直接回答。"}
    ]

# 渲染历史对话
for msg in st.session_state.messages:
    # 我们只给用户看 user 和 assistant 的文字，不渲染 tool 的底层数据
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content"):
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("输入你的问题（比如：集美大学考什么，或者 888*888 等）"):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 发起第一次大模型调用
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=st.session_state.messages,
                tools=tools,
                tool_choice="auto",
            )
            message = response.choices[0].message

            # 判断是否要调用工具
            if message.tool_calls:
                # 把模型的决定存入历史
                st.session_state.messages.append(message.model_dump())

                # 遍历要调用的工具
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    # 在网页上给个提示，让用户知道正在调用工具
                    tool_desc = "📚 正在查阅课程知识库..." if func_name == "search_courses" else "🧮 正在使用计算器..."
                    st.info(tool_desc)

                    # 执行工具
                    if func_name == "search_courses":
                        result = search_courses(args["query"])
                    elif func_name == "calculate":
                        result = calculate(args["expression"])

                    # 把结果存入历史
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": result
                    })

                # 发起第二次大模型调用（带着工具结果）
                st.info("🧠 正在整理最终回答...")
                second_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                )
                final_answer = second_response.choices[0].message.content

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            else:
                # 不需要工具，直接回答
                final_answer = message.content
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
