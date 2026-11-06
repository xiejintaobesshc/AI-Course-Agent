import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import glob
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ==========================================
# 配置
# ==========================================
DB_PATH = "./chroma_db"          # 向量库存放位置（会自动建文件夹）
COLLECTION_NAME = "courses"      # 集合，可以理解成"一张表"
KNOWLEDGE_DIR = "knowledge_base" # 你的课程 md 文件所在目录

# ==========================================
# 1. 初始化向量模型与 API
# ==========================================
print("正在加载本地向量模型...")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ==========================================
# 2. 打开（或创建）向量数据库
# ==========================================
# PersistentClient 会把数据持久化到硬盘，下次打开直接复用
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

# ==========================================
# 3. 首次运行才入库，之后直接复用（核心改进点！）
# ==========================================
if collection.count() == 0:
    print("第一次运行：读取文档 → 切块 → 计算向量 → 入库...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    documents, metadatas, ids = [], [], []
    for file_path in glob.glob(f"{KNOWLEDGE_DIR}/*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_name = os.path.basename(file_path)
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": file_name})   # 用 metadata 记录来源文件
            ids.append(f"{file_name}-{i}")

    # add 时会自动调用 embedding_fn 计算向量，不需要手动 encode
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ 已入库 {len(documents)} 个切块，下次启动会直接复用！")
else:
    print(f"✅ 检测到已有向量库（{collection.count()} 个切块），跳过入库，直接启动！")

# ==========================================
# 4. 检索并生成 (Retrieval & Generation)
# ==========================================
print("🤖 课程知识库助手已启动！(输入 'quit' 退出)\n")

while True:
    query = input("你提问: ")
    if query.lower() == "quit":
        break

    # 第一步：检索 —— 去向量库里找最相似的 Top 3
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 拼装上下文，附带来源文件名
    context_parts = [f"【文件名：{m['file']}】\n{d}" for d, m in zip(docs, metas)]
    context = "\n\n".join(context_parts)

    # 打印命中的来源，方便你判断检索质量
    print(f"\n📚 命中的来源文件：{[m['file'] for m in metas]}")

    # 第二步：生成 —— 拼 Prompt 发给 DeepSeek
    system_prompt = """你是一个专业的课程助手。请严格根据下面提供的【参考资料】来回答用户的问题。
如果参考资料里没有相关信息，请直接回答"根据现有资料，我无法回答这个问题"，绝对不要自己编造。

【参考资料】：
{context}"""

    print("思考中...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query},
        ],
    )

    print(f"\nAI 回答:\n{response.choices[0].message.content}")
    print("-" * 50)
