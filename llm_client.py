import json
import os
import sys
from typing import Any, Optional, Type

from litellm import completion
from pydantic import BaseModel


class LLMClient:

    def __init__(
        self,
        *,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str = "",
        max_retries: int = 6,
        reasoning_effort: str | None = None
    ):
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        if not self.api_key:
            raise ValueError("Azure API key not provided via args or AZURE_API_KEY.")
        self.api_base = api_base or os.getenv("AZURE_API_BASE")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._system_prompt = system_prompt
        self._max_retries = max_retries

    def __call__(
        self,
        instruction: str,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: str = ""
    ) -> Any:
        messages = []
        system_prompt = system_prompt or self._system_prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})

        try:
            extra_args: dict[str, Any] = {"num_retries": self._max_retries}
            if self.reasoning_effort:
                extra_args["reasoning_effort"] = self.reasoning_effort
            if response_format is not None:
                extra_args["response_format"] = {"type": "json_object"}

            if self.api_base:
                extra_args["api_base"] = self.api_base

            resp = completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                **extra_args
            )
            content = resp["choices"][0]["message"]["content"]
            if response_format is None:
                return content

            data = json.loads(content)
            if hasattr(response_format, "model_validate"):
                return response_format.model_validate(data)
            return response_format.parse_obj(data)
        except Exception as exc:
            print(f"LLMClient error: {exc}", file=sys.stderr)
            return None
