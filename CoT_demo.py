def get_cot_prompt(question):
    prompt = f"""Let's solve this step by step:
Question: {question}
Let's approach this step by step:
1)"""
    return prompt

# 使用示例
question = "If John has 5 apples and eats 2, then buys 3 more, how many apples does he have?"
prompt = get_cot_prompt(question)

# 使用语言模型获取回答
# response = llm(prompt)  # 需要接入具体的语言模型API
