from typing import Dict, List, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt

class GraphOfThought:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.thought_counter = 0
        self.final_answers: List[str] = []
        
    def add_thought(self, thought: str, parent_id: Optional[int] = None, score: float = 0.0) -> int:
        """
        Add a thought to the graph, optionally connecting it to a parent thought.
        Returns the ID of the new thought.
        """
        thought_id = self.thought_counter
        self.graph.add_node(thought_id, content=thought, score=score)
        
        if parent_id is not None and parent_id in self.graph:
            self.graph.add_edge(parent_id, thought_id)
            
        self.thought_counter += 1
        return thought_id
    
    def add_connection(self, from_id: int, to_id: int) -> bool:
        """Add a connection between two existing thoughts."""
        if from_id in self.graph and to_id in self.graph:
            self.graph.add_edge(from_id, to_id)
            return True
        return False
    
    def set_final_answer(self, answer: str, thought_id: Optional[int] = None):
        """Mark a thought as leading to a final answer."""
        self.final_answers.append(answer)
        if thought_id is not None:
            self.graph.nodes[thought_id]['is_final'] = True
    
    def get_thought_content(self, thought_id: int) -> Optional[str]:
        """Get the content of a specific thought."""
        if thought_id in self.graph:
            return self.graph.nodes[thought_id]['content']
        return None
    
    def get_children(self, thought_id: int) -> List[int]:
        """Get all direct children of a thought."""
        return list(self.graph.successors(thought_id))
    
    def get_parents(self, thought_id: int) -> List[int]:
        """Get all direct parents of a thought."""
        return list(self.graph.predecessors(thought_id))
    
    def get_all_paths(self) -> List[List[int]]:
        """Get all possible paths through the graph."""
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        all_paths = []
        
        for root in roots:
            paths = nx.all_simple_paths(self.graph, root, 
                                      [n for n in self.graph.nodes() 
                                       if self.graph.out_degree(n) == 0])
            all_paths.extend(list(paths))
        
        return all_paths
    
    def visualize(self, filename: Optional[str] = None):
        """
        Visualize the thought graph using networkx and matplotlib.
        Optionally save to a file if filename is provided.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=2000)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                             arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: f"ID: {node}\n{self.graph.nodes[node]['content'][:30]}..."
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Graph of Thoughts")
        if filename:
            plt.savefig(filename)
        plt.show()
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of all reasoning paths and final answers."""
        summary = []
        
        # Add all paths
        summary.append("=== Reasoning Paths ===")
        for path in self.get_all_paths():
            path_summary = []
            for thought_id in path:
                content = self.get_thought_content(thought_id)
                score = self.graph.nodes[thought_id].get('score', 0.0)
                path_summary.append(f"[{thought_id}] ({score:.2f}): {content}")
            summary.append(" -> ".join(path_summary))
        
        # Add final answers
        if self.final_answers:
            summary.append("\n=== Final Answers ===")
            for i, answer in enumerate(self.final_answers, 1):
                summary.append(f"{i}. {answer}")
        
        return "\n".join(summary)
    
    def clear(self):
        """Clear the graph and reset the thought counter."""
        self.graph.clear()
        self.thought_counter = 0
        self.final_answers = [] 