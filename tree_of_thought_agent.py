import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class TreeOfThoughtAgent:
    """
    一个实现 Tree of Thought 思考模式的 AI Agent
    允许模型探索多个思考路径，评估它们，并选择最佳路径
    """
    
    def __init__(self, name="思维树助手", model="deepseek-chat"):
        self.name = name
        self.model = model
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
    
    def explore_thoughts(self, question, num_branches=3):
        """
        探索多个思考分支
        
        Args:
            question: 用户问题
            num_branches: 要探索的思考分支数量
            
        Returns:
            生成的多个思考分支
        """
        messages = [
            {"role": "system", "content": f"""你是一个使用树状思维方式思考的AI助手。
面对问题时，请探索{num_branches}个不同的思考路径，每个路径代表一种不同的解决方案或思考角度。

对于每个思考路径，请遵循以下格式：

【思路{1}：简短标题】
- 初始想法：简要描述这个思考方向的起点
- 推理过程：
  1. 第一步推理...
  2. 第二步推理...
  3. 继续推理...
- 可能的结论：这个思路可能导向的结论
- 优点：列出这个思路的优点
- 缺点：列出这个思路的缺点
- 置信度：给这个思路一个0-100的置信度评分

请为所有{num_branches}个思路提供类似的分析。
最后，请评估所有思路，并选择最佳思路，解释你的选择理由。
"""},
            {"role": "user", "content": question}
        ]
        
        try:
            print(f"\n{self.name}正在探索多个思考路径...\n")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=3000
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="")
                    full_response += content
            
            return full_response
        except Exception as e:
            return f"思考过程中出现错误: {str(e)}"
    
    def deep_exploration(self, question, branch_to_explore):
        """
        深入探索特定的思考分支
        
        Args:
            question: 原始问题
            branch_to_explore: 要深入探索的分支编号或描述
            
        Returns:
            对特定分支的深入分析
        """
        messages = [
            {"role": "system", "content": f"""你是一个使用树状思维方式思考的AI助手。
现在，我希望你深入探索关于问题"{question}"的思路：{branch_to_explore}。

请按照以下步骤进行深入分析：

1. 【分支细化】将这个思路细分为3-5个子分支或子问题
2. 【详细推理】对每个子分支进行详细的推理分析
3. 【潜在障碍】识别每个子分支可能面临的障碍或挑战
4. 【解决方案】提出克服这些障碍的可能解决方案
5. 【整合结论】将所有子分支的分析整合成一个连贯的解决方案

请确保你的分析是深入的、全面的，并考虑到各种可能性和边缘情况。
"""},
            {"role": "user", "content": f"请深入分析思路：{branch_to_explore}"}
        ]
        
        try:
            print(f"\n{self.name}正在深入探索思路 {branch_to_explore}...\n")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=3000
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="")
                    full_response += content
            
            return full_response
        except Exception as e:
            return f"深入探索过程中出现错误: {str(e)}"
    
    def compare_and_decide(self, question, branches):
        """
        比较多个思考分支并做出决策
        
        Args:
            question: 原始问题
            branches: 要比较的分支列表或描述
            
        Returns:
            最终决策和理由
        """
        branches_str = "\n".join([f"- 思路{i+1}: {branch}" for i, branch in enumerate(branches)])
        
        messages = [
            {"role": "system", "content": """你是一个使用树状思维方式思考的AI助手。
现在，你需要比较多个思考路径，评估它们的优缺点，并做出最终决策。

请按照以下步骤进行：

1. 【评估标准】确定评估各个思路的关键标准（如可行性、效率、风险等）
2. 【比较分析】根据这些标准对每个思路进行评分和比较
3. 【权衡取舍】分析不同思路之间的权衡关系
4. 【最终决策】选择最佳思路或整合多个思路的优点
5. 【实施计划】提出如何实施这个决策的具体步骤

请确保你的决策过程是透明的、合理的，并充分考虑了所有相关因素。
"""},
            {"role": "user", "content": f"问题：{question}\n\n需要比较的思路：\n{branches_str}\n\n请比较这些思路并做出最终决策。"}
        ]
        
        try:
            print(f"\n{self.name}正在比较思路并做出决策...\n")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.5,  # 降低温度以获得更确定的决策
                max_tokens=2000
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="")
                    full_response += content
            
            return full_response
        except Exception as e:
            return f"决策过程中出现错误: {str(e)}"
    
    def run_interactive(self):
        """
        运行交互式会话
        """
        print(f"{self.name}已启动。输入'退出'结束会话。")
        
        while True:
            question = input("\n请输入你的问题: ")
            
            if question.lower() in ['退出', 'exit', 'quit']:
                print(f"{self.name}: 再见！")
                break
            
            # 第一阶段：探索多个思考路径
            branches_response = self.explore_thoughts(question)
            
            # 询问用户是否要深入探索特定分支
            explore_further = input("\n是否要深入探索某个特定思路？(输入思路编号或描述，或输入'否'跳过): ")
            
            if explore_further.lower() not in ['否', 'no', 'n']:
                # 第二阶段：深入探索特定分支
                self.deep_exploration(question, explore_further)
            
            # 询问用户是否要比较多个分支并做出决策
            compare_decision = input("\n是否要比较多个思路并做出决策？(是/否): ")
            
            if compare_decision.lower() in ['是', 'yes', 'y']:
                branches = []
                print("请输入要比较的思路描述（每行一个，输入空行结束）：")
                while True:
                    branch = input()
                    if not branch:
                        break
                    branches.append(branch)
                
                if branches:
                    # 第三阶段：比较分支并做出决策
                    self.compare_and_decide(question, branches)
            
            print("\n思考过程完成！")

# 使用示例
if __name__ == "__main__":
    agent = TreeOfThoughtAgent()
    agent.run_interactive() 