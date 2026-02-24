from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    def add(self, other: UsageTotals) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls


@dataclass
class UsageTracker:
    overall: UsageTotals = field(default_factory=UsageTotals)
    by_model: Dict[str, UsageTotals] = field(default_factory=dict)

    def record_from_litellm(self, kwargs: Dict[str, Any], completion_response: Any) -> None:
        usage = {}
        try:
            usage = completion_response.get("usage", {}) or {}
        except Exception:
            usage = {}

        totals = UsageTotals(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
            calls=1,
        )

        model = str(kwargs.get("model") or "unknown")
        self.overall.add(totals)
        bucket = self.by_model.setdefault(model, UsageTotals())
        bucket.add(totals)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Optimization Cost Report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- calls: {self.overall.calls}")
        lines.append(f"- prompt_tokens: {self.overall.prompt_tokens}")
        lines.append(f"- completion_tokens: {self.overall.completion_tokens}")
        lines.append(f"- total_tokens: {self.overall.total_tokens}")
        if self.overall.cached_tokens:
            lines.append(f"- cached_tokens: {self.overall.cached_tokens}")
        if self.overall.reasoning_tokens:
            lines.append(f"- reasoning_tokens: {self.overall.reasoning_tokens}")
        lines.append("")

        lines.append("## By Model")
        lines.append("")
        if not self.by_model:
            lines.append("- No model usage recorded.")
            return "\n".join(lines)

        for model, totals in sorted(self.by_model.items()):
            lines.append(f"- {model}")
            lines.append(f"- calls: {totals.calls}")
            lines.append(f"- prompt_tokens: {totals.prompt_tokens}")
            lines.append(f"- completion_tokens: {totals.completion_tokens}")
            lines.append(f"- total_tokens: {totals.total_tokens}")
            if totals.cached_tokens:
                lines.append(f"- cached_tokens: {totals.cached_tokens}")
            if totals.reasoning_tokens:
                lines.append(f"- reasoning_tokens: {totals.reasoning_tokens}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def write_markdown(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        return path
