# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()  # 加载 .env 文件
# 通过 Prompt 设计实现：
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

### 通过 Prompt 设计实现：
messages = [
    {"role": "system", "content": """请按以下格式回答问题：
思考过程：
1. 首先...
2. 然后...
3. 接着...
4. 最后...
5. 最终答案：...

最终答案：..."""},
    {"role": "user", "content": "解释一下信用卡和储蓄卡的区别"}
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


#使用函数调用方式（更结构化）
'''
# 定义函数格式
functions = [{
    "name": "solve_problem",
    "description": "解决问题并展示思考过程",
    "parameters": {
        "type": "object",
        "properties": {
            "thinking_steps": {
                "type": "array",
                "description": "思考的步骤",
                "items": {"type": "string"}
            },
            "final_answer": {
                "type": "string",
                "description": "最终答案"
            }
        },
        "required": ["thinking_steps", "final_answer"]
    }
}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "计算 23 + 45 是多少?"}],
    tools=[{
        "type": "function",
        "function": functions[0]
    }],
    tool_choice={"type": "function", "function": {"name": "solve_problem"}}
)

# 解析返回的 JSON
result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

# 输出思考步骤
print("思考过程：")
for i, step in enumerate(result["thinking_steps"], 1):
    print(f"{i}. {step}")

print("\n最终答案：")
print(result["final_answer"])
'''

# 使用函数调用方式更复杂的思考链（更结构化）
'''
def think_step_by_step(question):
    messages = [
        {"role": "system", "content": """你是一个会仔细思考的助手。
解决问题时，请：
1. 分析问题的关键点
2. 列出可能的解决方案
3. 评估每个方案的优缺点
4. 选择最佳方案并说明原因
5. 给出最终答案

请用"【分析】"、"【方案】"、"【评估】"、"【选择】"、"【答案】"等标记清晰地分隔每个步骤。"""},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

# 使用示例
question = "我有一个重要的会议和一个朋友的生日聚会时间冲突了，我该怎么办？"
think_step_by_step(question)
'''
