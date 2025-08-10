import re
from typing import Dict, Any

class HeuristicRouter:
    """
    Implements heuristic-based cost-aware routing.
    Classifies prompts based on length and keywords to select the optimal model.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.complex_keywords = ["analyze", "solve", "debug", "explain in depth", "optimization", "algorithm", "step-by-step"]
        self.length_threshold = config.get("length_threshold", 150) # tokens approx.
        self.default_model_easy = config.get("easy_model", "gpt-3.5-turbo")
        self.default_model_complex = config.get("complex_model", "gpt-4")
        self.code_pattern = re.compile(r"```[\s\S]*?```|def\s+\w+\s*\(|\b(class|import|for|while)\b")

    def estimate_complexity(self, prompt: str) -> str:
        """Estimates complexity as 'Easy' or 'Complex'."""

        # 1. Check for code blocks
        if self.code_pattern.search(prompt):
            return "Complex"

        # 2. Check for complex keywords
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in self.complex_keywords):
            return "Complex"

        # 3. Check length (simple approximation of token count by word count)
        if len(prompt.split()) > self.length_threshold:
            return "Complex"

        return "Easy"

    def route(self, prompt: str) -> str:
        """Routes the prompt to the appropriate model."""
        complexity = self.estimate_complexity(prompt)
        if complexity == "Complex":
            return self.default_model_complex
        else:
            return self.default_model_easy