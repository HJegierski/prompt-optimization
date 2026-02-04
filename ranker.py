from typing import Optional

from pydantic import BaseModel

from llm_client import LLMClient
from models import Preference, Product


class RankingResponse(BaseModel):
    explanation: str
    result: Preference


class Ranker:

    def __init__(
        self,
        prompt_template: str,
        client: Optional[LLMClient] = None,
    ):
        if client is None:
            raise ValueError("Ranker requires an LLMClient instance.")
        self.prompt_template = prompt_template
        self._client = client

    def _format_prompt(self, query: str, product_lhs: Product, product_rhs: Product) -> str:
        return self.prompt_template.format(
            query=query,
            product_lhs=product_lhs,
            product_rhs=product_rhs,
        )

    def rank(
        self,
        query: str,
        product_lhs: Product,
        product_rhs: Product,
    ) -> Optional[Preference]:
        """
        Returns a Preference or None.
        """
        prompt = self._format_prompt(query, product_lhs, product_rhs)
        response = self._client(prompt, response_format=RankingResponse)
        if isinstance(response, RankingResponse):
            return response.result
        return None
