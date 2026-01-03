import os

from llm_client import LLMClient
from optimizers import LLMOptimizer, OptimizerConfig, DSPyOptimizer


def run_llm_optimizer():
    client = LLMClient(model="gpt-5", reasoning_effort="high")
    optimizer = LLMOptimizer(client)

    seed_prompt_path = "prompts/brain_prompt.txt"
    with open(seed_prompt_path, "r", encoding="utf-8") as f:
        seed_prompt = f.read()

    task_context = """
You are comparing two candidate products (LHS, RHS) for a given user query using the WANDS product search relevance dataset.
WANDS provides human judgments with three relevance grades: Exact, Partial, Irrelevant (used in our code as 2, 1, 0).
The goal is to decide which product is more relevant to the query or return Neither when they are equally relevant or insufficiently supported.

Return a JSON object with:
- result: "LHS" | "RHS" | "Neither"
- explanation: one sentence citing the decisive attributes or the lack thereof.
"""

    prompt_path = optimizer.optimize_and_save(
        seed_prompt=seed_prompt,
        save_as_strategy="llm_prompt",
        task_context=task_context,
        system_prompt_override=None,
        directory="prompts",
        write_notes=True
    )
    print("Optimized prompt saved to:", prompt_path)


def run_dspy_optimizer():
    cfg = OptimizerConfig(
        task_model="azure/gpt-5-nano",
        reflection_model="azure/gpt-5",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        save_as_strategy="gepa_prompt",
        temperature=1.0,
        max_tokens=16_000,
        sample_size=150,
        test_size=100,
        auto_budget="light",
        max_metric_calls=None,
    )

    optimizer = DSPyOptimizer(cfg)
    prompt_path = optimizer.optimize_and_save()
    print("Optimized prompt saved to:", prompt_path)


if __name__ == "__main__":
    run_llm_optimizer()
    # run_dspy_optimizer()
    pass
