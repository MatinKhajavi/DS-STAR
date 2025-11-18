from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message to the LLM and return the response.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The LLM's response as a string
        """
        pass

    def extract_code_block(self, response: str) -> str:
        """
        Extract code from markdown code blocks.

        Args:
            response: LLM response potentially containing code blocks

        Returns:
            Extracted code or original response if no code block found
        """
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()

        return response.strip()

