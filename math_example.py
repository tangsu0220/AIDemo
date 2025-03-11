from chain_of_thought import ChainOfThought
from tree_of_thought import TreeOfThought
from graph_of_thought import GraphOfThought
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def solve_with_chain_of_thought():
    print("\n=== Chain of Thought Approach ===")
    cot = ChainOfThought()
    
    # Solving step by step linearly
    cot.add_thought("First, let's organize the prices from highest to lowest:")
    cot.add_thought("$25, $22, $20, $18, $15, $12, $10")
    
    cot.add_thought("We can group items into sets of 3 to maximize savings:")
    cot.add_thought("Group 1: $25, $22, $20 (50% off $20)")
    cot.add_thought("Group 2: $18, $15, $12 (50% off $12)")
    cot.add_thought("Remaining: $10")
    
    cot.add_thought("Calculate Group 1: $25 + $22 + ($20 × 0.5) = $57")
    cot.add_thought("Calculate Group 2: $18 + $15 + ($12 × 0.5) = $39")
    cot.add_thought("Add remaining $10")
    
    cot.set_final_answer("Total cost: $57 + $39 + $10 = $106")
    
    print(cot.get_reasoning_chain())

def solve_with_tree_of_thought():
    print("\n=== Tree of Thought Approach ===")
    tot_solver = TreeOfThought(
        model="deepseek-chat",
        api_type="deepseek",
        max_tokens=200,
        temperature=0.7,
        n_candidates=4,
        max_steps=4,
        max_branches=3,
        verbose=True
    )

    problem = """
    A store has a sale where if you buy 2 items, you get the third item at 50% off. 
    The discount applies to the least expensive item among the three.
    If you buy 7 items with costs $10, $12, $15, $18, $20, $22, and $25, how much will you pay in total?
    """

    evaluation_prompt = """
    Problem: {problem}

    Reasoning step: {thought}

    Evaluate how promising this reasoning step is for solving the math problem on a scale of 1-10.
    Consider:
    - Is the mathematical reasoning correct?
    - Does it correctly apply the discount rules?
    - Is it making progress toward the final calculation?
    - Are there any calculation errors?

    Score (1-10):
    """

    solution = tot_solver.solve(problem, evaluation_prompt)
    
    print("\n=== SOLUTION ===")
    print(f"Final answer: {solution['final_answer']}")
    print(f"Final score: {solution['final_score']:.2f}")
    print("\nReasoning path:")
    for i, thought in enumerate(solution['thoughts']):
        print(f"Step {i+1} (Score: {solution['scores'][i]:.2f}): {thought}")

def solve_with_graph_of_thought():
    print("\n=== Graph of Thought Approach ===")
    got = GraphOfThought()
    
    # Initial approach considerations
    root = got.add_thought("How to optimize the grouping of 7 items for maximum savings?")
    
    # Branch 1: Sort by price
    sort_branch = got.add_thought("Sort items by price: $25, $22, $20, $18, $15, $12, $10", parent_id=root)
    
    # Branch 2: Different grouping strategies
    strategy_branch = got.add_thought("Consider different grouping strategies", parent_id=root)
    
    # Explore grouping strategies
    strategy1 = got.add_thought("Strategy 1: Group highest prices together", parent_id=strategy_branch, score=0.9)
    got.add_thought("Group 1: $25, $22, $20 (save $10 on $20)\nGroup 2: $18, $15, $12 (save $6 on $12)\nRemaining: $10", 
                   parent_id=strategy1)
    
    strategy2 = got.add_thought("Strategy 2: Mix high and low prices", parent_id=strategy_branch, score=0.7)
    got.add_thought("Group 1: $25, $20, $10 (save $5 on $10)\nGroup 2: $22, $18, $12 (save $6 on $12)\nRemaining: $15",
                   parent_id=strategy2)
    
    # Calculations for best strategy
    calc1 = got.add_thought("Calculate Strategy 1:\n$25 + $22 + $10 + $18 + $15 + $6 + $10 = $106", 
                           parent_id=strategy1, score=0.95)
    calc2 = got.add_thought("Calculate Strategy 2:\n$25 + $20 + $5 + $22 + $18 + $6 + $15 = $111",
                           parent_id=strategy2, score=0.85)
    
    # Final conclusion
    conclusion = got.add_thought("Strategy 1 yields the lowest total cost", parent_id=calc1)
    got.set_final_answer("Total cost: $106", conclusion)
    
    # Visualize the reasoning process
    got.visualize("math_reasoning_graph.png")
    print(got.get_reasoning_summary())

if __name__ == "__main__":
    # Solve using all three approaches
    solve_with_chain_of_thought()
    solve_with_tree_of_thought()
    solve_with_graph_of_thought() 