from typing import Optional, Sequence
from pydantic import BaseModel, Field
import os

from llm_client import LLMClient


class OptimizedPrompt(BaseModel):
    """Structured result for prompt optimization."""
    optimized_prompt: str = Field(..., description="The improved prompt template.")
    rationale: str = Field(..., description="Brief reasoning about the changes.")
    changelog: Sequence[str] = Field(default_factory=list, description="Bulleted list of edits.")


class LLMOptimizer:

    SYSTEM_TASK = (
        "You are a prompt engineer. Your job is to rewrite prompts for a SMALL/NON-FRONTIER model. "
        "Requirements:\n"
        "• Keep the output format of the downstream task stable.\n"
        "• Be concise and robust to noisy inputs.\n"
        "• Prefer explicit step-by-step checks over flowery language.\n"
        "• Include guardrails: if required context is missing, ask for it or abstain.\n"
        "Return only JSON matching the response schema."
    )

    def __init__(self, client: LLMClient):
        self._client = client

    def optimize(
        self,
        seed_prompt: str,
        *,
        task_context: str = "",
        system_prompt_override: Optional[str] = None,
    ) -> OptimizedPrompt:

        user_instruction = f"""
You are given a SEED PROMPT that will be used by a small LLM.

TASK CONTEXT:
{task_context or 'N/A'}

SEED PROMPT (verbatim):
---
{seed_prompt}
---

Rewrite the prompt. Preserve any variables or placeholders. Provide a very short rationale and a bullet changelog.
"""

        parsed = self._client(
            user_instruction,
            response_format=OptimizedPrompt,
            system_prompt=system_prompt_override or self.SYSTEM_TASK,
        )
        if isinstance(parsed, OptimizedPrompt):
            return parsed

        return OptimizedPrompt(
            optimized_prompt=seed_prompt,
            rationale="Parsing failed; returning original prompt.",
            changelog=[],
        )

    def save_prompt(self, prompt_text: str, save_as_strategy: str, directory: str = "prompts") -> str:
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory, f"{save_as_strategy}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        return out_path

    def optimize_and_save(
            self,
            seed_prompt: str,
            save_as_strategy: str = "brain_prompt_llm",
            task_context: str = "",
            system_prompt_override: Optional[str] = None,
            directory: str = "prompts",
            write_notes: bool = False,
    ) -> str:
        """
        End-to-end: run LLM-based optimization and save the optimized prompt file. Returns the path.
        """
        result = self.optimize(
            seed_prompt,
            task_context=task_context,
            system_prompt_override=system_prompt_override,
        )
        prompt_path = self.save_prompt(
            result.optimized_prompt,
            save_as_strategy=save_as_strategy,
            directory=directory,
        )

        if write_notes:
            notes_path = os.path.join(directory, f"{save_as_strategy}_notes.md")
            try:
                with open(notes_path, "w", encoding="utf-8") as f:
                    f.write(f"# Optimization Notes: {save_as_strategy}\n\n")
                    f.write(f"**Rationale**\n\n{result.rationale}\n\n")
                    if result.changelog:
                        f.write("**Changelog**\n\n")
                        for item in result.changelog:
                            f.write(f"- {item}\n")
                        f.write("\n")
            except Exception:
                pass

        return prompt_path
