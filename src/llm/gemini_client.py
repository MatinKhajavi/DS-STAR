import os
from typing import Optional
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from src.llm.base import BaseLLMClient

load_dotenv()


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models using the new google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
    ):
        if genai is None:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )

        super().__init__(model, api_key)

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message to Gemini

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The model's response
        """
        try:
            if system_prompt:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

