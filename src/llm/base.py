from abc import ABC, abstractmethod
from typing import Optional
import inspect


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.log_dir: Optional[str] = None
        self.query_id: Optional[str] = None
        self._prompt_counter = 0
    
    def set_logging_context(self, log_dir: Optional[str], query_id: Optional[str]):
        """Set logging context for prompt logging."""
        self.log_dir = log_dir
        self.query_id = query_id
        self._prompt_counter = 0
    
    def _log_prompt(self, prompt: str, system_prompt: Optional[str] = None):
        """Log prompt to file if logging is enabled."""
        if not self.log_dir:
            return
        
        try:
            from src.utils.logging_utils import log_prompt as save_prompt
            from pathlib import Path
            
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                caller_file = Path(caller_frame.f_code.co_filename).stem
                caller_func = caller_frame.f_code.co_name
                agent_name = f"{caller_file}"
                step = caller_func
            else:
                agent_name = "unknown"
                step = "unknown"
            
            full_prompt = ""
            if system_prompt:
                full_prompt += f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
            full_prompt += f"=== USER PROMPT ===\n{prompt}"
            
            self._prompt_counter += 1
            step_with_counter = f"{step}_{self._prompt_counter:02d}"
            
            save_prompt(full_prompt, agent_name, step_with_counter, self.log_dir, self.query_id)
        except Exception as e:
            # Don't fail if logging fails
            pass

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

