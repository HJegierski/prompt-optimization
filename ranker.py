from typing import Any, Dict, Optional, Union

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

    def _coerce_product(self, product: Union[Product, Dict[str, Any]]) -> Product:
        if isinstance(product, Product):
            return product
        return Product(
            id=str(product.get("id", "")),
            name=str(product.get("name", "")),
            description=str(product.get("description", "")),
            class_name=str(product.get("class_name", "")),
            category_hierarchy=str(product.get("category_hierarchy", "")),
            grade=int(product.get("grade", 0) or 0),
        )

    def _format_prompt(self, query: str, product_lhs: Union[Product, Dict[str, Any]], product_rhs: Union[Product, Dict[str, Any]]) -> str:
        lhs = self._coerce_product(product_lhs)
        rhs = self._coerce_product(product_rhs)
        return self.prompt_template.format(
            query=query,
            product_lhs=lhs,
            product_rhs=rhs,
        )

    def rank(
        self,
        query: str,
        product_lhs: Union[Product, Dict[str, Any]],
        product_rhs: Union[Product, Dict[str, Any]],
    ) -> Optional[Preference]:
        """
        Returns a Preference or None.
        """
        prompt = self._format_prompt(query, product_lhs, product_rhs)
        response = self._client(prompt, response_format=RankingResponse)
        if isinstance(response, RankingResponse):
            return response.result
        return None
