import os
from typing import Any, Optional, Type

from openai import AzureOpenAI
from pydantic import BaseModel


class AzureOpenAIClient:

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        deployment: str = "gpt-5-nano",
        system_prompt: str = "",
        max_retries: int = 6,
        reasoning_effort: str | None = None
    ):
        """Thin wrapper around Azure OpenAI chat + structured output parsing."""
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_API_BASE")
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        if not self.azure_endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint/key not provided via args or environment.")
        self.api_version = api_version or os.getenv("AZURE_API_VERSION")
        self.deployment = deployment
        self.reasoning_effort = reasoning_effort
        self._system_prompt = system_prompt
        self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                max_retries=max_retries
            )

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
            if response_format is None:
                resp = self._client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort
                )
            else:
                resp = self._client.beta.chat.completions.parse(
                    model=self.deployment,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    response_format=response_format
                )

            choice = resp.choices[0]
            msg = choice.message
            return msg.parsed if hasattr(msg, "parsed") else msg
        except Exception as e:
            return None
