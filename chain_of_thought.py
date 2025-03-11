class ChainOfThought:
    def __init__(self):
        self.thoughts = []
        self.final_answer = None

    def add_thought(self, thought):
        """Add a step in the reasoning process."""
        self.thoughts.append(thought)
    
    def set_final_answer(self, answer):
        """Set the final conclusion after the chain of reasoning."""
        self.final_answer = answer
    
    def get_reasoning_chain(self):
        """Get the complete chain of reasoning."""
        chain = "\n".join([f"Step {i+1}: {thought}" for i, thought in enumerate(self.thoughts)])
        if self.final_answer:
            chain += f"\n\nFinal Answer: {self.final_answer}"
        return chain

    def clear(self):
        """Clear the current chain of thoughts."""
        self.thoughts = []
        self.final_answer = None 