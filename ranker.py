from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel

from azure_openai_client import AzureOpenAIClient


class RankingResponse(BaseModel):
    explanation: str
    result: Literal["LHS", "RHS", "Neither"]


class Ranker:

    def __init__(
        self,
        prompt_template: str,
        client: Optional[AzureOpenAIClient] = None,
    ):
        if client is None:
            raise ValueError("Ranker requires an AzureOpenAIClient instance.")
        self.prompt_template = prompt_template
        self._client = client

    def _format_prompt(self, query: str, product_lhs: Dict[str, Any], product_rhs: Dict[str, Any]) -> str:
        return self.prompt_template.format(
            query=query,
            product_lhs_name=product_lhs.get("name", ""),
            product_lhs_description=product_lhs.get("description", ""),
            product_rhs_name=product_rhs.get("name", ""),
            product_rhs_description=product_rhs.get("description", ""),
        )

    def rank(
        self,
        query: str,
        product_lhs: Dict[str, Any],
        product_rhs: Dict[str, Any],
    ) -> Optional[Literal["LHS", "RHS", "Neither"]]:
        """
        Returns preference string or None.
        """
        prompt = self._format_prompt(query, product_lhs, product_rhs)
        response = self._client(prompt, response_format=RankingResponse)
        if isinstance(response, RankingResponse):
            return response.result
        return None
