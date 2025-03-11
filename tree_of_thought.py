import os
import re
import time
import numpy as np
from typing import List, Dict, Callable, Tuple, Any, Optional
import requests
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class TreeOfThought:
    """
    Implementation of the Tree of Thought approach for problem-solving with language models.
    This allows exploring multiple reasoning paths to find better solutions.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_type: str = "openai",  # "openai" or "deepseek"
        max_tokens: int = 150,
        temperature: float = 1.0,
        n_candidates: int = 3,
        max_steps: int = 5,
        max_branches: int = 3,
        verbose: bool = False,
        stream: bool = True  # Added streaming parameter
    ):
        """
        Initialize the Tree of Thought solver.
        
        Args:
            model: Model to use for generation
            api_type: Type of API to use ("openai" or "deepseek")
            max_tokens: Maximum tokens for each generation
            temperature: Temperature for generation (higher = more diverse)
            n_candidates: Number of candidate thoughts to generate at each step
            max_steps: Maximum number of reasoning steps
            max_branches: Maximum branches to explore at each step
            verbose: Whether to print detailed progress
            stream: Whether to stream the model's thinking process
        """
        self.api_type = api_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.max_steps = max_steps
        self.max_branches = max_branches
        self.verbose = verbose
        self.stream = stream
        
        # Load API keys from environment variables
        if api_type == "openai":
            import openai
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai.api_key = self.openai_api_key
            self.client = openai
        elif api_type == "deepseek":
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def generate_thoughts(self, prompt: str, n: int = 3) -> List[str]:
        """
        Generate n possible next thoughts given the prompt.
        
        Args:
            prompt: The prompt to generate thoughts from
            n: Number of thoughts to generate
            
        Returns:
            List of generated thoughts
        """
        try:
            if self.api_type == "openai":
                return self._generate_thoughts_openai(prompt, n)
            elif self.api_type == "deepseek":
                return self._generate_thoughts_deepseek(prompt, n)
        except Exception as e:
            print(f"Error generating thoughts: {e}")
            # Return some basic fallback thoughts if API fails
            return [f"Let's think about this problem step {i+1}..." for i in range(n)]
    
    def _generate_thoughts_openai(self, prompt: str, n: int) -> List[str]:
        """Generate thoughts using OpenAI API"""
        thoughts = []
        
        for i in range(n):
            if self.verbose:
                print(f"\nGenerating thought {i+1}/{n}...")
                
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates distinct reasoning steps."},
                    {"role": "user", "content": f"{prompt}\n\nGenerate a next step in the reasoning process:"}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=self.stream
            )
            
            if self.stream:
                # Handle streaming response
                thought = ""
                print(f"Thought {i+1}: ", end="", flush=True)
                for chunk in response:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            print(content, end="", flush=True)
                            thought += content
                print()  # New line after streaming completes
                thoughts.append(thought.strip())
            else:
                # Handle non-streaming response
                thought = response.choices[0].message.content.strip()
                thoughts.append(thought)
                if self.verbose:
                    print(f"Thought {i+1}: {thought[:100]}...")
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        return thoughts
    
    def _generate_thoughts_deepseek(self, prompt: str, n: int) -> List[str]:
        """Generate thoughts using DeepSeek API"""
        thoughts = []
        
        for i in range(n):
            if self.verbose:
                print(f"\nGenerating thought {i+1}/{n}...")
                
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates distinct reasoning steps."},
                    {"role": "user", "content": f"{prompt}\n\nGenerate a next step in the reasoning process:"}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": self.stream
            }
            
            if self.stream:
                # Handle streaming for DeepSeek
                response = requests.post(
                    self.deepseek_api_url, 
                    headers=self.headers, 
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                
                thought = ""
                print(f"Thought {i+1}: ", end="", flush=True)
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                        content = chunk['choices'][0]['delta']['content']
                                        if content:
                                            print(content, end="", flush=True)
                                            thought += content
                            except json.JSONDecodeError:
                                pass
                
                print()  # New line after streaming completes
                thoughts.append(thought.strip())
            else:
                # Handle non-streaming for DeepSeek
                response = requests.post(self.deepseek_api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                thought = result["choices"][0]["message"]["content"].strip()
                thoughts.append(thought)
                if self.verbose:
                    print(f"Thought {i+1}: {thought[:100]}...")
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        return thoughts
    
    def evaluate_thoughts(self, thoughts: List[str], problem: str, evaluation_prompt: str) -> List[float]:
        """
        Evaluate the quality/promise of each thought for solving the problem.
        
        Args:
            thoughts: List of thoughts to evaluate
            problem: The original problem
            evaluation_prompt: Prompt template for evaluation
            
        Returns:
            List of scores for each thought (higher is better)
        """
        scores = []
        
        for i, thought in enumerate(thoughts):
            try:
                if self.verbose:
                    print(f"\nEvaluating thought {i+1}/{len(thoughts)}...")
                    
                prompt = evaluation_prompt.format(
                    problem=problem,
                    thought=thought
                )
                
                if self.api_type == "openai":
                    score = self._evaluate_thought_openai(prompt, thought, i+1)
                elif self.api_type == "deepseek":
                    score = self._evaluate_thought_deepseek(prompt, thought, i+1)
                
                scores.append(score)
                
                if self.verbose and not self.stream:
                    print(f"Thought: {thought[:50]}...\nScore: {score:.2f}")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error evaluating thought: {e}")
                scores.append(0.5)  # Default neutral score
        
        return scores
    
    def _evaluate_thought_openai(self, prompt: str, thought: str, thought_num: int) -> float:
        """Evaluate a thought using OpenAI API"""
        if self.stream:
            print(f"Evaluating thought {thought_num}: ", end="", flush=True)
            
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates the quality of reasoning steps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.2,  # Low temperature for more consistent evaluations
                stream=self.stream
            )
            
            response_text = ""
            for chunk in response:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        response_text += content
            print()  # New line after streaming completes
        else:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates the quality of reasoning steps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.2  # Low temperature for more consistent evaluations
            )
            response_text = response.choices[0].message.content.strip()
        
        score = self._parse_evaluation_score(response_text)
        if self.stream:
            print(f"Score: {score:.2f}")
        return score
    
    def _evaluate_thought_deepseek(self, prompt: str, thought: str, thought_num: int) -> float:
        """Evaluate a thought using DeepSeek API"""
        import json
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that evaluates the quality of reasoning steps."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.2,
            "stream": self.stream
        }
        
        if self.stream:
            print(f"Evaluating thought {thought_num}: ", end="", flush=True)
            
            response = requests.post(
                self.deepseek_api_url, 
                headers=self.headers, 
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            response_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                    content = chunk['choices'][0]['delta']['content']
                                    if content:
                                        print(content, end="", flush=True)
                                        response_text += content
                        except json.JSONDecodeError:
                            pass
            
            print()  # New line after streaming completes
        else:
            response = requests.post(self.deepseek_api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
        
        score = self._parse_evaluation_score(response_text)
        if self.stream:
            print(f"Score: {score:.2f}")
        return score
    
    def _parse_evaluation_score(self, response_text: str) -> float:
        """Parse evaluation score from response text"""
        try:
            # Look for patterns like "Score: 7/10" or just numbers
            score_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\/\s*10)?', response_text)
            if score_matches:
                # If it looks like a score out of 10, normalize to 0-1
                score = float(score_matches[0])
                if score > 10:  # Probably not a 0-10 score
                    score = min(score, 10) / 10
                else:
                    score = score / 10
            else:
                # Fallback: count positive words
                positive_words = ['good', 'great', 'excellent', 'promising', 'helpful', 'useful', 'correct']
                negative_words = ['bad', 'poor', 'wrong', 'incorrect', 'irrelevant', 'confusing']
                
                score = 0.5  # Default neutral score
                for word in positive_words:
                    if word in response_text.lower():
                        score += 0.1
                for word in negative_words:
                    if word in response_text.lower():
                        score -= 0.1
                
                score = max(0.1, min(score, 1.0))  # Clamp between 0.1 and 1.0
        except:
            score = 0.5  # Default score if parsing fails
            
        return score
    
    def solve(self, problem: str, evaluation_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a problem using Tree of Thought approach.
        
        Args:
            problem: The problem to solve
            evaluation_prompt: Custom prompt for evaluation (optional)
            
        Returns:
            Dictionary with the solution and metadata
        """
        if evaluation_prompt is None:
            evaluation_prompt = """
            Problem: {problem}
            
            Reasoning step: {thought}
            
            Evaluate how promising this reasoning step is for solving the problem on a scale of 1-10.
            Consider:
            - Is this a logical next step?
            - Does it make progress toward the solution?
            - Is it relevant to the problem?
            
            Provide your score (1-10):
            """
        
        # Start with the problem as the initial prompt
        current_paths = [{"prompt": problem, "thoughts": [], "scores": []}]
        best_solution = {"prompt": problem, "thoughts": [], "scores": [], "final_answer": ""}
        
        for step in range(self.max_steps):
            if self.verbose:
                print(f"\n--- Step {step+1}/{self.max_steps} ---")
            
            new_paths = []
            
            for path_idx, path in enumerate(current_paths):
                if self.verbose:
                    print(f"\nExploring path {path_idx+1}/{len(current_paths)}")
                
                current_prompt = path["prompt"]
                
                # Generate candidate next thoughts
                new_thoughts = self.generate_thoughts(current_prompt, self.n_candidates)
                
                # Evaluate the thoughts
                scores = self.evaluate_thoughts(new_thoughts, problem, evaluation_prompt)
                
                # Create new paths with each thought
                for thought, score in zip(new_thoughts, scores):
                    new_prompt = f"{current_prompt}\n\nThought: {thought}"
                    new_path = {
                        "prompt": new_prompt,
                        "thoughts": path["thoughts"] + [thought],
                        "scores": path["scores"] + [score]
                    }
                    new_paths.append(new_path)
            
            # Sort paths by the average score and keep the top k
            new_paths.sort(key=lambda x: sum(x["scores"]) / len(x["scores"]) if x["scores"] else 0, reverse=True)
            current_paths = new_paths[:self.max_branches]
            
            if self.verbose:
                print(f"\nExploring {len(current_paths)} paths after step {step+1}")
                for i, path in enumerate(current_paths):
                    avg_score = sum(path["scores"]) / len(path["scores"]) if path["scores"] else 0
                    print(f"Path {i+1}: Avg score = {avg_score:.2f}, Last thought: {path['thoughts'][-1][:50]}...")
        
        # After all steps, generate final answers for each path
        final_answers = []
        final_scores = []
        
        if self.verbose:
            print("\n--- Generating Final Answers ---")
        
        for path_idx, path in enumerate(current_paths):
            try:
                if self.verbose:
                    print(f"\nGenerating final answer for path {path_idx+1}/{len(current_paths)}")
                
                final_prompt = f"{path['prompt']}\n\nBased on this reasoning, what is the final answer to the original problem?"
                
                if self.api_type == "openai":
                    final_answer = self._generate_final_answer_openai(final_prompt, path_idx+1)
                elif self.api_type == "deepseek":
                    final_answer = self._generate_final_answer_deepseek(final_prompt, path_idx+1)
                
                final_answers.append(final_answer)
                
                # Evaluate the final answer
                eval_prompt = f"""
                Problem: {problem}
                
                Reasoning process and final answer:
                {path['prompt']}
                
                Final answer: {final_answer}
                
                On a scale of 1-10, how correct and complete is this answer?
                """
                
                if self.verbose:
                    print(f"\nEvaluating final answer for path {path_idx+1}/{len(current_paths)}")
                
                if self.api_type == "openai":
                    score = self._evaluate_final_answer_openai(eval_prompt, path_idx+1)
                elif self.api_type == "deepseek":
                    score = self._evaluate_final_answer_deepseek(eval_prompt, path_idx+1)
                
                final_scores.append(score)
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating final answer: {e}")
                final_answers.append("Failed to generate a final answer.")
                final_scores.append(0.0)
        
        # Select the best final answer
        if final_scores:
            best_idx = np.argmax(final_scores)
            best_path = current_paths[best_idx]
            best_solution = {
                "prompt": best_path["prompt"],
                "thoughts": best_path["thoughts"],
                "scores": best_path["scores"],
                "final_answer": final_answers[best_idx],
                "final_score": final_scores[best_idx],
                "all_final_answers": final_answers,
                "all_final_scores": final_scores
            }
            
            if self.verbose:
                print(f"\nSelected best answer from path {best_idx+1} with score {final_scores[best_idx]:.2f}")
        
        return best_solution
    
    def _generate_final_answer_openai(self, final_prompt: str, path_idx: int) -> str:
        """Generate final answer using OpenAI API"""
        import json
        
        if self.stream:
            print(f"Generating final answer for path {path_idx}: ", end="", flush=True)
            
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides final answers based on a reasoning process."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more focused final answers
                stream=self.stream
            )
            
            final_answer = ""
            for chunk in response:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        final_answer += content
            print()  # New line after streaming completes
            return final_answer.strip()
        else:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides final answers based on a reasoning process."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3  # Lower temperature for more focused final answers
            )
            
            return response.choices[0].message.content.strip()
    
    def _generate_final_answer_deepseek(self, final_prompt: str, path_idx: int) -> str:
        """Generate final answer using DeepSeek API"""
        import json
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides final answers based on a reasoning process."},
                {"role": "user", "content": final_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
            "stream": self.stream
        }
        
        if self.stream:
            print(f"Generating final answer for path {path_idx}: ", end="", flush=True)
            
            response = requests.post(
                self.deepseek_api_url, 
                headers=self.headers, 
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            final_answer = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                    content = chunk['choices'][0]['delta']['content']
                                    if content:
                                        print(content, end="", flush=True)
                                        final_answer += content
                        except json.JSONDecodeError:
                            pass
            
            print()  # New line after streaming completes
            return final_answer.strip()
        else:
            response = requests.post(self.deepseek_api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
    
    def _evaluate_final_answer_openai(self, eval_prompt: str, path_idx: int) -> float:
        """Evaluate final answer using OpenAI API"""
        if self.stream:
            print(f"Evaluating final answer for path {path_idx}: ", end="", flush=True)
            
            eval_response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates the quality of solutions."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=50,
                temperature=0.2,
                stream=self.stream
            )
            
            eval_text = ""
            for chunk in eval_response:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        eval_text += content
            print()  # New line after streaming completes
        else:
            eval_response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates the quality of solutions."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=50,
                temperature=0.2
            )
            
            eval_text = eval_response.choices[0].message.content.strip()
        
        score = self._parse_evaluation_score(eval_text)
        if self.stream:
            print(f"Final score: {score:.2f}")
        return score
    
    def _evaluate_final_answer_deepseek(self, eval_prompt: str, path_idx: int) -> float:
        """Evaluate final answer using DeepSeek API"""
        import json
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that evaluates the quality of solutions."},
                {"role": "user", "content": eval_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.2,
            "stream": self.stream
        }
        
        if self.stream:
            print(f"Evaluating final answer for path {path_idx}: ", end="", flush=True)
            
            response = requests.post(
                self.deepseek_api_url, 
                headers=self.headers, 
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            eval_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                    content = chunk['choices'][0]['delta']['content']
                                    if content:
                                        print(content, end="", flush=True)
                                        eval_text += content
                        except json.JSONDecodeError:
                            pass
            
            print()  # New line after streaming completes
        else:
            response = requests.post(self.deepseek_api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            eval_text = result["choices"][0]["message"]["content"].strip()
        
        score = self._parse_evaluation_score(eval_text)
        if self.stream:
            print(f"Final score: {score:.2f}")
        return score