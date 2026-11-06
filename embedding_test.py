import os
# 强制走国内镜像下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("正在加载本地中文向量模型 (第一次运行会下载约 100MB)...")
# 加载智源研究院的轻量级中文向量模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 1. 准备我们的语料库（假装是从课程大纲里切出来的几句话）
sentences = [
    "集美大学的人工智能课程包含机器学习基础。",
    "微软公开课分为 Azure 云服务和 AI 基础两大部分。",
    "今天天气真不错，适合去海边玩。",
    "深度学习里的卷积神经网络常用于图像识别。"
]

# 2. 把文字变成向量
print("正在计算向量...")
embeddings = model.encode(sentences)

print(f"✅ 转换成功！每个句子变成了 {embeddings.shape[1]} 维的数字向量。\n")

# 3. 模拟用户提问
query = "集美大学的 AI 课学什么？"
query_embedding = model.encode([query])

# 4. 计算用户提问与每一句话的“余弦相似度”
# 余弦相似度越接近 1，代表两句话语义越接近
similarities = cosine_similarity(query_embedding, embeddings)[0]

# 5. 把结果打印出来看看
print(f"用户提问: {query}\n")
for i, sentence in enumerate(sentences):
    print(f"文本: {sentence}")
    print(f"相似度得分: {similarities[i]:.4f}")
    print("-" * 30)