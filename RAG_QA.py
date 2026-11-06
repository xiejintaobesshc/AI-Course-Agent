import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import glob
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ==========================================
# 1. 初始化模型与 API
# ==========================================
print("正在加载本地向量模型...")
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ==========================================
# 2. 读取文档与切块 (Chunking)
# ==========================================
print("正在读取 knowledge_base 文件夹里的文档...")
all_texts = []

# 读取文件夹下所有的 .md 文件
for file_path in glob.glob("knowledge_base/*.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 记录一下这段文字来自哪个文件，方便溯源
        file_name = os.path.basename(file_path)
        all_texts.append(f"【文件名：{file_name}】\n{content}")

# 切块：因为文件太长，模型吃不下，我们需要把它切碎
# chunk_size=500 表示每 500 个字切一刀，chunk_overlap=50 表示前后留 50 个字的重叠，防止语义断裂
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.create_documents(all_texts)
chunk_contents = [chunk.page_content for chunk in chunks]
print(f"✅ 文档处理完毕，被切分成了 {len(chunk_contents)} 个小块。")

# ==========================================
# 3. 将所有切块变成向量 (Embedding)
# ==========================================
print("正在把知识块变成向量，建立索引（这需要几秒钟）...")
chunk_embeddings = embedding_model.encode(chunk_contents)
print("✅ 向量索引建立完成！\n")

# ==========================================
# 4. 检索并生成 (Retrieval & Generation)
# ==========================================
print("🤖 课程知识库助手已启动！(输入 'quit' 退出)")

while True:
    query = input("\n你提问: ")
    if query.lower() == 'quit':
        break

    # 第一步：把问题变成向量
    query_embedding = embedding_model.encode([query])

    # 第二步：去知识库里找最相似的 Top 3
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    # argsort 返回从小到大排序的索引，[-3:] 取最后 3 个（最大的），[::-1] 倒序排
    top3_indices = similarities.argsort()[-3:][::-1]

    # 取出这 3 段话
    context = "\n\n".join([chunk_contents[i] for i in top3_indices])

    # 第三步：拼装 Prompt，发给 DeepSeek
    system_prompt = """你是一个专业的课程助手。请严格根据下面提供的【参考资料】来回答用户的问题。
如果参考资料里没有相关信息，请直接回答“根据现有资料，我无法回答这个问题”，绝对不要自己编造。

【参考资料】：
{context}"""

    print("思考中...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query}
        ]
    )

    print(f"\nAI 回答:\n{response.choices[0].message.content}")
    print("-" * 50)
