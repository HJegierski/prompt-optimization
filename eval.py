"""Evaluation loop and metrics for prompt ranking strategies."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Union

import pandas as pd

from llm_client import LLMClient
from models import Preference, Product
from ranker import Ranker
from wands_data import pairwise_split

PRODUCT_FIELD_MAP = {
    "id": "product_id",
    "name": "product_name",
    "description": "product_description",
    "class_name": "product_class",
    "category_hierarchy": "category hierarchy",
    "grade": "grade",
}


def build_product(row: pd.Series, side: str) -> Product:
    if side not in ("x", "y"):
        raise ValueError(f"side must be 'x' or 'y', got {side}")
    values = {key: row[f"{column}_{side}"] for key, column in PRODUCT_FIELD_MAP.items()}
    values["grade"] = int(values["grade"])
    return Product(**values)


def output_row(
    query: str,
    product_lhs: Product,
    product_rhs: Product,
    human_preference: Union[Preference, str],
    agent_preference: Union[Preference, str],
) -> Dict[str, object]:
    human_value = human_preference.value if isinstance(human_preference, Preference) else human_preference
    agent_value = agent_preference.value if isinstance(agent_preference, Preference) else agent_preference
    return {
        'query': query,
        'product_name_lhs': product_lhs.name,
        'product_description_lhs': product_lhs.description,
        'product_id_lhs': product_lhs.id,
        'product_class_lhs': product_lhs.class_name,
        'category_hierarchy_lhs': product_lhs.category_hierarchy,
        'grade_lhs': product_lhs.grade,
        'product_name_rhs': product_rhs.name,
        'product_description_rhs': product_rhs.description,
        'product_id_rhs': product_rhs.id,
        'product_class_rhs': product_rhs.class_name,
        'category_hierarchy_rhs': product_rhs.category_hierarchy,
        'grade_rhs': product_rhs.grade,
        'human_preference': human_value,
        'agent_preference': agent_value
    }


def _extract_grade(product: Union[Product, Dict[str, int]]) -> int:
    if isinstance(product, Product):
        return int(product.grade)
    return int(product["grade"])


def compare_products(product_lhs: Union[Product, Dict[str, int]], product_rhs: Union[Product, Dict[str, int]]) -> Preference:
    preference = _extract_grade(product_lhs) - _extract_grade(product_rhs)
    if preference > 0:
        return Preference.LHS
    elif preference < 0:
        return Preference.RHS
    return Preference.NEITHER


@dataclass(frozen=True)
class Metrics:
    total: int
    same: int
    different: int
    no_pref: int
    agent_has_pref: int
    overall_agreement: float
    accuracy_on_human_pref: float
    coverage: float
    selective_precision: float
    selective_recall: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "same": self.same,
            "different": self.different,
            "no_pref": self.no_pref,
            "agent_has_pref": self.agent_has_pref,
            "overall_agreement": self.overall_agreement,
            "accuracy_on_human_pref": self.accuracy_on_human_pref,
            "coverage": self.coverage,
            "selective_precision": self.selective_precision,
            "selective_recall": self.selective_recall,
        }


def results_df_stats(results_df: pd.DataFrame) -> None:
    metrics = compute_metrics(results_df)
    if metrics.total == 0:
        print("No results yet.")
        return
    print(f"Same Preference: {metrics.same}, Different Preference: {metrics.different}, No Preference: {metrics.no_pref}")
    print(
        f"Agree: {metrics.overall_agreement:.1f}% | Acc(human-pref): {metrics.accuracy_on_human_pref:.1f}% | "
        f"Coverage: {metrics.coverage:.1f}% | SelPrec: {metrics.selective_precision:.1f}% | "
        f"SelRec: {metrics.selective_recall:.1f}%"
    )


def has_been_labeled(results_df: pd.DataFrame, query: str, product_lhs: Product, product_rhs: Product) -> bool:
    result_exists = (len(results_df) > 0
                     and (results_df[(results_df['query'] == query) &
                          (results_df['product_id_lhs'] == product_lhs.id) &
                          (results_df['product_id_rhs'] == product_rhs.id)].shape[0] > 0))
    return result_exists


def compute_metrics(results_df: pd.DataFrame) -> Metrics:
    total = len(results_df)
    if total == 0:
        return Metrics(
            total=0,
            same=0,
            different=0,
            no_pref=0,
            agent_has_pref=0,
            overall_agreement=0.0,
            accuracy_on_human_pref=0.0,
            coverage=0.0,
            selective_precision=0.0,
            selective_recall=0.0,
        )

    agent = results_df['agent_preference']
    human = results_df['human_preference']

    is_agent_neither = agent.eq(Preference.NEITHER.value)
    is_human_neither = human.eq(Preference.NEITHER.value)
    agent_pref = ~is_agent_neither
    human_pref = ~is_human_neither
    agree = agent.eq(human)

    same_preference = int(agree.sum())
    different_preference = int(((agent != human) & agent_pref & human_pref).sum())
    no_preference = int(is_agent_neither.sum())
    agent_has_preference = int(agent_pref.sum())

    overall_agreement = agree.mean() * 100
    accuracy_on_human_pref = ((agree & human_pref).sum() / human_pref.sum() * 100) if human_pref.any() else 0.0
    coverage = agent_pref.mean() * 100
    correct_when_agent_pref = (agree & agent_pref & human_pref).sum()
    selective_precision = (correct_when_agent_pref / agent_pref.sum() * 100) if agent_pref.any() else 0.0
    selective_recall = (correct_when_agent_pref / human_pref.sum() * 100) if human_pref.any() else 0.0

    return Metrics(
        total=total,
        same=same_preference,
        different=different_preference,
        no_pref=no_preference,
        agent_has_pref=agent_has_preference,
        overall_agreement=overall_agreement,
        accuracy_on_human_pref=accuracy_on_human_pref,
        coverage=coverage,
        selective_precision=selective_precision,
        selective_recall=selective_recall
    )


def results_df_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    return compute_metrics(results_df).to_dict()


def load_prompt(strategy: str) -> str:
    prompt_path = os.path.join('prompts', f'{strategy}.txt')
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_cached_results(pickle_path: str, destroy_cache: bool) -> pd.DataFrame:
    if destroy_cache and os.path.exists(pickle_path):
        os.remove(pickle_path)
    try:
        return pd.read_pickle(pickle_path)
    except FileNotFoundError:
        return pd.DataFrame()


def save_results(results_df: pd.DataFrame, pickle_path: str, csv_path: str) -> None:
    try:
        results_df.to_pickle(pickle_path)
    except Exception as exc:
        print(f"Warning: failed to save cache ({exc}).")
    try:
        results_df.to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"Warning: failed to save results ({exc}).")


def run_strategy(
        strategy: str,
        sample_size=500,
        test_size: int = 400,
        destroy_cache=False,
        seed=42,
        output_dir: str = "data/eval",
        verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    os.makedirs(output_dir, exist_ok=True)
    _, test_df = pairwise_split(sample_size, test_size=test_size, seed=seed)
    df = test_df
    run_name = strategy.replace(" ", "_")
    pickle_path = os.path.join(output_dir, f'{run_name}.pkl')
    csv_path = os.path.join(output_dir, f'{run_name}_results.csv')
    txt_path = os.path.join(output_dir, f'{run_name}_summary.txt')

    results_df = load_cached_results(pickle_path, destroy_cache)
    prompt_template = load_prompt(strategy)

    client = LLMClient()
    ranker = Ranker(prompt_template=prompt_template, client=client)

    for _, row in df.iterrows():
        query = row['query_x']
        product_lhs = build_product(row, "x")
        product_rhs = build_product(row, "y")
        if has_been_labeled(results_df, query, product_lhs, product_rhs):
            continue
        ground_truth = compare_products(product_lhs, product_rhs)
        prediction = ranker.rank(query, product_lhs, product_rhs) or Preference.NEITHER
        results_df = pd.concat([results_df, pd.DataFrame([output_row(
            query,
            product_lhs,
            product_rhs,
            ground_truth,
            prediction
        )])],
                               ignore_index=True)
        save_results(results_df, pickle_path, csv_path)
        if verbose:
            results_df_stats(results_df)

    metrics = results_df_metrics(results_df)
    summary_lines = [
        f"Strategy: {strategy}",
        f"Total: {metrics['total']}",
        f"Overall agreement: {metrics['overall_agreement']:.2f}%",
        f"Accuracy (human-pref): {metrics['accuracy_on_human_pref']:.2f}%",
        f"Coverage: {metrics['coverage']:.2f}%",
        f"Selective precision: {metrics['selective_precision']:.2f}%",
        f"Selective recall: {metrics['selective_recall']:.2f}%",
        f"Same: {metrics['same']} | Different: {metrics['different']} | No Pref: {metrics['no_pref']}",
        f"Timestamp: {datetime.utcnow().isoformat()}Z"
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    return results_df, metrics


def evaluate_strategies(
        strategies: List[str],
        sample_size=500,
        test_size=400,
        destroy_cache=False,
        seed=42,
        output_dir: str = "data/eval",
        verbose: bool = True
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    for s in strategies:
        _, m = run_strategy(s, sample_size, test_size, destroy_cache, seed, output_dir, verbose)
        m['strategy'] = s
        all_metrics.append(m)
    metrics_df = pd.DataFrame(all_metrics).sort_values(by="overall_agreement", ascending=False)
    summary_csv = os.path.join(output_dir, "strategies_summary.csv")
    summary_txt = os.path.join(output_dir, "strategies_summary.txt")
    metrics_df.to_csv(summary_csv, index=False)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Strategy Evaluation Summary\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        for _, r in metrics_df.iterrows():
            f.write(
                f"{r.strategy}: Overall={r.overall_agreement:.2f}% | Acc(human-pref)={r.accuracy_on_human_pref:.2f}% "
                f"| Coverage={r.coverage:.2f}% | SelPrec={r.selective_precision:.2f}% | SelRec={r.selective_recall:.2f}% "
                f"| Same={r.same} Diff={r.different} NoPref={r.no_pref} Total={r.total}\n"
            )
    return metrics_df


def main(
        strategy: Union[str, List[str]],
        sample_size: int = 500,
        test_size: int = 400,
        destroy_cache: bool = False,
        seed: int = 42,
        output_dir: str = "data/eval",
        verbose: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(strategy, (list, tuple)):
        return evaluate_strategies(list(strategy), sample_size, test_size, destroy_cache, seed, output_dir, verbose)
    else:
        results_df, metrics = run_strategy(strategy, sample_size, test_size, destroy_cache, seed, output_dir, verbose)
        if verbose:
            results_df_stats(results_df)
        return metrics


if __name__ == "__main__":
    main(
        strategy=["brain_prompt", "llm_prompt", "gepa_prompt"],
        sample_size=250,
        test_size=100
    )
