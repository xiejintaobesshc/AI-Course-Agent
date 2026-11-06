import os

# 强制离线
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import ast
import operator
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ==========================================
# 1. 页面与资源初始化
# ==========================================
st.set_page_config(page_title="ReAct 智能课程助手", page_icon="🧠", layout="centered")
st.title("🧠 ReAct 智能课程助手")
st.caption("具备多轮记忆 + 连续工具调用的完整智能体")


@st.cache_resource
def init_resources():
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="courses", embedding_function=embedding_fn)
    return client, collection


client, collection = init_resources()


# ==========================================
# 2. 定义工具
# ==========================================
def search_courses(query):
    results = collection.query(query_texts=[query], n_results=3)
    return "\n\n".join([f"【{m['file']}】\n{d}" for d, m in zip(results["documents"][0], results["metadatas"][0])])


def calculate(expression):
    # 用 AST 白名单替代 eval：只放行数字和基本运算符，杜绝任意代码执行
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval_node(node.operand))
        raise ValueError("只支持数字和 + - * / // % ** 运算")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"计算错误: {e}"


tools = [
    {"type": "function", "function": {"name": "search_courses", "description": "查询课程大纲、内容等问题时调用",
                                      "parameters": {"type": "object", "properties": {
                                          "query": {"type": "string", "description": "搜索关键词"}},
                                                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "进行数学计算时调用", "parameters": {"type": "object",
                                                                                                      "properties": {
                                                                                                          "expression": {
                                                                                                              "type": "string",
                                                                                                              "description": "数学表达式"}},
                                                                                                      "required": [
                                                                                                          "expression"]}}}
]

# ==========================================
# 3. 聊天界面与历史记录 (记忆)
# ==========================================
# 每次页面的初始状态，系统提示词一定要给足，约束好它的行为
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个智能课程助手。你可以查阅知识库，也可以做计算。请根据用户需求自主决定使用什么工具，如果遇到复杂任务，可以多次调用。"}
    ]

# 渲染历史对话 (只渲染文本，不渲染内部工具调用过程)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content"):
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ==========================================
# 4. ReAct 核心循环 (核心逻辑)
# ==========================================
if prompt := st.chat_input("输入你的问题（测试多轮记忆和复杂任务）"):
    # 记录用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 这是循环的起点，只要模型还在调工具，就继续转
        max_iterations = 5  # 防止死循环，设个上限
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            with st.spinner(f"思考中... (第 {iteration} 轮)"):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                    tools=tools,
                    tool_choice="auto",
                )
                message = response.choices[0].message

            # 【核心判断】如果模型决定不调工具了，说明它准备好最终回答了
            if not message.tool_calls:
                final_answer = message.content
                st.markdown(final_answer)
                # 把最终回答存入历史
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                break  # 退出循环

            # 如果模型要调工具
            st.info(f"🤖 模型决定调用工具 (第 {iteration} 轮)")
            # 必须把模型的要求存入历史（否则下次请求会报错）
            st.session_state.messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                tool_desc = "📚 查阅知识库" if func_name == "search_courses" else "🧮 使用计算器"
                st.info(f"🔧 正在执行: {tool_desc} - 参数: {args}")

                # 执行工具
                if func_name == "search_courses":
                    result = search_courses(args["query"])
                elif func_name == "calculate":
                    result = calculate(args["expression"])

                # 把工具执行结果存入历史
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result
                })

            st.info("🔄 结果已返回，模型继续思考...")
            # 循环继续，带着工具结果再次发给模型
