import os

# 护身符：强制离线，禁止联网检查，秒加载本地缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# 1. 初始化大模型
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 2. 初始化向量数据库（为了 RAG）
print("正在加载向量库...")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="courses", embedding_function=embedding_fn)
print("向量库加载成功！\n")


# 3. 定义工具（供模型调用）
def search_courses(query):
    """去向量库里查资料"""
    print(f"  [后台执行] 正在检索知识库: {query}")
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return "\n\n".join([f"【文件名：{m['file']}】\n{d}" for d, m in zip(docs, metas)])


def calculate(expression):
    """算数学题"""
    print(f"  [后台执行] 正在计算: {expression}")
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# 4. 告诉模型我们有哪些工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "当用户询问课程大纲、课程内容、授课老师等知识库相关问题时，调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
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
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，比如 '85+90'"}
                },
                "required": ["expression"],
            },
        }
    }
]

# 5. 运行测试
user_input = input("你提问: ")
messages = [{"role": "user", "content": user_input}]

print("\n--- 第一次交互：模型决策 ---")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message

# 6. 判断模型是否需要工具
if message.tool_calls:
    print(f"🤖 模型决定调用工具！")
    messages.append(message)  # 把模型的决策加进去

    # 有可能模型同时调用了多个工具
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # 执行对应的函数
        if func_name == "search_courses":
            result = search_courses(args["query"])
        elif func_name == "calculate":
            result = calculate(args["expression"])

        # 把结果加进历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    print("\n--- 第二次交互：生成最终回答 ---")
    second_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    print(f"\n✨ 最终回答:\n{second_response.choices[0].message.content}")
else:
    print(f"\n✨ 直接回答:\n{message.content}")
