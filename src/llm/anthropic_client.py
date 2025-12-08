import os
from typing import Optional
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from src.llm.base import BaseLLMClient

load_dotenv()


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
    ):
        if Anthropic is None:
            raise ImportError("anthropic is not installed. Install it with: pip install anthropic")

        super().__init__(model, api_key)

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=api_key)

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message to Claude.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The model's response
        """
        self._log_prompt(prompt, system_prompt)
        
        kwargs = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

