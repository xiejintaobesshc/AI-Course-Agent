from openai import OpenAI

# 1. 初始化客户端（换上你的 Key）
client = OpenAI(
    api_key="sk-ba7787f6cf98437a9d0947070a9aff3c",  # 注意保留引号
    base_url="https://api.deepseek.com",
)

# 2. 发起对话请求
print("正在呼叫 DeepSeek 超级大脑...\n")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个精通深度学习的 AI 导师，说话简明扼要。"},
        {"role": "user", "content": "我刚学完了 ResNet 和 BERT，现在开始学调用你们大模型 API，你觉得我下一步该学什么？"}
    ]
)

# 3. 打印回答
print(response.choices[0].message.content)
