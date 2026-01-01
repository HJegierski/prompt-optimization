import os
import dataclasses
from typing import List, Optional, Tuple, Dict, Any

import dspy
import pandas as pd

from wands_data import pairwise_split
from eval import compare_products


class PairwiseRankSignature(dspy.Signature):
    """
    You are a product ranking judge.

    Task: Given a shopping query and two candidate products (LHS and RHS), decide which one
    better satisfies the query strictly by *relevance*, not popularity or price.

    Output rules:
    - result MUST be one of: "LHS", "RHS", or "Neither".
    - explanation should be concise, citing specific attributes from each product.

    Constraints:
    - Prefer exact matches over partial.
    - If both are unrelated to the query, choose "Neither".
    - If one is a partial match and the other is irrelevant, prefer the partial match.
    - Ignore marketing fluff; focus on the core attributes the query implies.
    """
    query: str = dspy.InputField(desc="User search query")
    product_lhs_name: str = dspy.InputField()
    product_lhs_description: str = dspy.InputField()
    product_rhs_name: str = dspy.InputField()
    product_rhs_description: str = dspy.InputField()

    explanation: str = dspy.OutputField(desc="Short justification")
    result: str = dspy.OutputField(desc='One of "LHS", "RHS", "Neither"')


class ProductRanker(dspy.Module):
    """
    Minimal DSPy program: one predictor with an instruction that GEPA will evolve.
    """
    def __init__(self, name: str = "ranker"):
        super().__init__()
        self.name = name
        self.rank = dspy.Predict(PairwiseRankSignature)

    def forward(
        self,
        query: str,
        product_lhs_name: str,
        product_lhs_description: str,
        product_rhs_name: str,
        product_rhs_description: str
    ):
        return self.rank(
            query=query,
            product_lhs_name=product_lhs_name,
            product_lhs_description=product_lhs_description,
            product_rhs_name=product_rhs_name,
            product_rhs_description=product_rhs_description
        )


@dataclasses.dataclass
class OptimizerConfig:
    task_model: str = "azure/gpt-5-nano"        # student LM (cheap, many trials)
    reflection_model: str = "azure/gpt-5"       # stronger LM for GEPA reflection

    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None

    temperature: float = 1.0
    max_tokens: int = 1_000

    # GEPA budget knobs
    auto_budget: str = "light"                  # "light" | "medium" | "heavy"
    max_metric_calls: Optional[int] = None
    reflection_minibatch_size: int = 3
    candidate_selection: str = "pareto"         # "pareto" or "current_best"

    # Data
    sample_size: int = 600
    test_size: int = 100
    seed: int = 42

    save_as_strategy: str = "gepa_prompt"


class DSPyOptimizer:

    def __init__(self, cfg: OptimizerConfig = OptimizerConfig()):
        self.cfg = cfg

        lm_kwargs = dict(
            model_type="chat",
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if self.cfg.api_key:
            lm_kwargs["api_key"] = self.cfg.api_key
        if self.cfg.api_base:
            lm_kwargs["api_base"] = self.cfg.api_base
        if self.cfg.api_version:
            lm_kwargs["api_version"] = self.cfg.api_version

        task_lm = dspy.LM(self.cfg.task_model, **lm_kwargs)
        reflection_lm = dspy.LM(self.cfg.reflection_model, **lm_kwargs)

        dspy.settings.configure(lm=task_lm)
        self.reflection_lm = reflection_lm

        self.program = ProductRanker()

        self._optimized_program: Optional[ProductRanker] = None
        self._trainset: List[dspy.Example] = []
        self._valset: List[dspy.Example] = []

    def _to_examples(self, df: pd.DataFrame) -> List[dspy.Example]:
        """
        Create examples with inputs matching signature and a 'gold' label `result`.
        """
        examples: List[dspy.Example] = []
        for _, row in df.iterrows():
            query = row["query_x"]
            lhs_name = row["product_name_x"]
            lhs_desc = row["product_description_x"]
            rhs_name = row["product_name_y"]
            rhs_desc = row["product_description_y"]

            gold_result = compare_products(
                {
                    "grade": row["grade_x"]
                },
                {
                    "grade": row["grade_y"]
                }
            )

            ex = dspy.Example(
                query=query,
                product_lhs_name=lhs_name,
                product_lhs_description=lhs_desc,
                product_rhs_name=rhs_name,
                product_rhs_description=rhs_desc,
                result=gold_result
            ).with_inputs(
                "query",
                "product_lhs_name",
                "product_lhs_description",
                "product_rhs_name",
                "product_rhs_description",
            )
            examples.append(ex)
        return examples

    def _load_data(self) -> Tuple[List[dspy.Example], List[dspy.Example]]:
        train_df, val_df = pairwise_split(
            sample_size=self.cfg.sample_size,
            test_size=self.cfg.test_size,
            seed=self.cfg.seed
        )
        n = len(val_df) // 2
        return self._to_examples(val_df.iloc[:n]), self._to_examples(val_df.iloc[n:])

    @staticmethod
    def _metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float | Dict[str, Any]:
        predicted = (getattr(pred, "result", None) or "").strip()
        gold_label = (getattr(gold, "result", None) or "").strip()

        score = 1.0 if predicted == gold_label else 0.0

        return score
        # feedback = (
        #     f'Predicted "{predicted}" but gold is "{gold_label}". '
        #     "Prefer exact category/attribute matches to query; pick Neither if both are irrelevant; "
        #     "if one partial vs. one irrelevant, select the partial match. "
        #     "Use concrete product attributes from the input to justify."
        # )
        # return {"score": score, "feedback": feedback}

    def optimize(self) -> ProductRanker:
        """
        Runs GEPA and stores the optimized program.
        """
        if not self._trainset or not self._valset:
            self._trainset, self._valset = self._load_data()

        gepa = dspy.GEPA(
            metric=self._metric,
            auto=self.cfg.auto_budget,
            max_metric_calls=self.cfg.max_metric_calls,
            reflection_minibatch_size=self.cfg.reflection_minibatch_size,
            candidate_selection_strategy=self.cfg.candidate_selection,
            reflection_lm=self.reflection_lm,
            track_stats=True,
        )

        self._optimized_program = gepa.compile(
            self.program,
            trainset=self._trainset,
            valset=self._valset
        )
        return self._optimized_program

    def _build_ranker_prompt_from_instruction(self, instruction_text: str) -> str:
        """
        Convert the optimized instruction to your Ranker prompt template with your variables.
        The Ranker expects placeholders: {query}, {product_lhs_*}, {product_rhs_*}
        and returns a pydantic-parseable JSON with fields matching RankingResponse.
        """
        return f"""\
{instruction_text}

You will receive a shopping query and two candidate products (LHS and RHS).

<INPUT>
Query: {{query}}

[LHS]
Name: {{product_lhs_name}}
Description: {{product_lhs_description}}

[RHS]
Name: {{product_rhs_name}}
Description: {{product_rhs_description}}
</INPUT>

<OUTPUT FORMAT>
Return ONLY valid JSON with exactly these fields:
{{{{
  "explanation": "short, specific justification comparing attributes to the query",
  "result": "LHS" | "RHS" | "Neither"
}}}}
</OUTPUT FORMAT>
"""

    def _extract_instruction_text(self, prog: ProductRanker) -> str:
        """
        Best-effort extraction of the optimized instruction for the 'rank' predictor.
        Falls back to the signature docstring if needed.
        """
        try:
            sig = prog.rank.signature
            text = getattr(sig, "instructions", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception as e:
            pass

        doc = PairwiseRankSignature.__doc__ or ""
        return doc.strip()

    def save_prompt(self, instruction_text: str, directory: str = "prompts") -> str:
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory or ".", f"{self.cfg.save_as_strategy}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self._build_ranker_prompt_from_instruction(instruction_text))
        return out_path

    def optimize_and_save(self, directory: str = "prompts") -> str:
        """
        End-to-end: run GEPA and save the optimized prompt file. Returns the path.
        """
        prog = self.optimize()
        instruction_text = self._extract_instruction_text(prog)
        return self.save_prompt(instruction_text, directory)
