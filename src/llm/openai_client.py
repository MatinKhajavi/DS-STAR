import os
from typing import Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.llm.base import BaseLLMClient

load_dotenv()


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models (GPT-4, GPT-5, etc.)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("openai is not installed. Install it with: pip install openai")

        super().__init__(model, api_key)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message to OpenAI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The model's response
        """
        self._log_prompt(prompt, system_prompt)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

