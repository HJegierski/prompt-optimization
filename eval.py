import os
import pandas as pd
from ranker import Ranker
from azure_openai_client import AzureOpenAIClient
from wands_data import pairwise_df, train_test_split
from datetime import datetime
from typing import List, Dict, Tuple, Union


def product_row_to_dict(row):
    if 'product_name_x' in row:
        return {
            'id': row['product_id_x'],
            'name': row['product_name_x'],
            'description': row['product_description_x'],
            'class': row['product_class_x'],
            'category_hierarchy': row['category hierarchy_x'],
            'grade': row['grade_x']
        }
    elif 'product_name_y' in row:
        return {
            'id': row['product_id_y'],
            'name': row['product_name_y'],
            'description': row['product_description_y'],
            'class': row['product_class_y'],
            'category_hierarchy': row['category hierarchy_y'],
            'grade': row['grade_y']
        }


def output_row(query, product_lhs, product_rhs, human_preference, agent_preference):
    return {
        'query': query,
        'product_name_lhs': product_lhs['name'],
        'product_description_lhs': product_lhs['description'],
        'product_id_lhs': product_lhs['id'],
        'product_class_lhs': product_lhs['class'],
        'category_hierarchy_lhs': product_lhs['category_hierarchy'],
        'grade_lhs': product_lhs['grade'],
        'product_name_rhs': product_rhs['name'],
        'product_description_rhs': product_rhs['description'],
        'product_id_rhs': product_rhs['id'],
        'product_class_rhs': product_rhs['class'],
        'category_hierarchy_rhs': product_rhs['category_hierarchy'],
        'grade_rhs': product_rhs['grade'],
        'human_preference': human_preference,
        'agent_preference': agent_preference
    }


def compare_products(product_lhs, product_rhs):
    preference = product_lhs['grade'] - product_rhs['grade']
    if preference > 0:
        return 'LHS'
    elif preference < 0:
        return 'RHS'
    else:
        return 'Neither'


def results_df_stats(results_df):
    total = len(results_df)
    if total == 0:
        print("No results yet.")
        pass

    agent = results_df['agent_preference']
    human = results_df['human_preference']

    is_agent_neither = agent.eq('Neither')
    is_human_neither = human.eq('Neither')
    agent_pref = ~is_agent_neither  # agent predicted LHS/RHS
    human_pref = ~is_human_neither  # human labeled LHS/RHS
    agree = agent.eq(human)

    # 1) Overall agreement (includes Neither==Neither)
    overall_agreement = agree.mean() * 100

    # 2) Accuracy when human has a preference (ignore human==Neither)
    accuracy_on_human_pref = ((agree & human_pref).sum() / human_pref.sum() * 100) if human_pref.any() else 0.0

    # 3) Coverage: agent predicts LHS/RHS
    coverage = agent_pref.mean() * 100

    # 4) Selective precision: correct given agent predicted LHS/RHS (and human had a pref)
    correct_when_agent_pref = (agree & agent_pref & human_pref).sum()
    selective_precision = (correct_when_agent_pref / agent_pref.sum() * 100) if agent_pref.any() else 0.0

    # 5) Selective recall: correct among human LHS/RHS cases (penalizes agent abstains)
    selective_recall = (correct_when_agent_pref / human_pref.sum() * 100) if human_pref.any() else 0.0

    same_preference = int(agree.sum())
    different_preference = int(((agent != human) & agent_pref & human_pref).sum())  # only LHS/RHS disagreements
    no_preference = int(is_agent_neither.sum())

    print(f"Same Preference: {same_preference}, Different Preference: {different_preference}, No Preference: {no_preference}")
    print(
        f"Agree: {overall_agreement:.1f}% | Acc(human-pref): {accuracy_on_human_pref:.1f}% | "
        f"Coverage: {coverage:.1f}% | SelPrec: {selective_precision:.1f}% | "
        f"SelRec: {selective_recall:.1f}%"
    )


def has_been_labeled(results_df, query, product_lhs, product_rhs):
    result_exists = (len(results_df) > 0
                     and (results_df[(results_df['query'] == query) &
                          (results_df['product_id_lhs'] == product_lhs['id']) &
                          (results_df['product_id_rhs'] == product_rhs['id'])].shape[0] > 0))
    return result_exists


def results_df_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    total = len(results_df)
    if total == 0:
        return dict(
            total=0, same=0, different=0, no_pref=0, agent_has_pref=0,
            overall_agreement=0.0, accuracy_on_human_pref=0.0,
            coverage=0.0, selective_precision=0.0, selective_recall=0.0
        )

    agent = results_df['agent_preference']
    human = results_df['human_preference']

    is_agent_neither = agent.eq('Neither')
    is_human_neither = human.eq('Neither')
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

    return dict(
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
    combined_df = pairwise_df(sample_size + test_size, seed)
    train_df, test_df = train_test_split(combined_df, test_size=test_size, seed=seed)
    df = test_df
    results_df = pd.DataFrame()
    run_name = strategy.replace(" ", "_")
    pickle_path = os.path.join(output_dir, f'{run_name}.pkl')
    csv_path = os.path.join(output_dir, f'{run_name}_results.csv')
    txt_path = os.path.join(output_dir, f'{run_name}_summary.txt')

    if destroy_cache and os.path.exists(pickle_path):
        os.remove(pickle_path)
    try:
        results_df = pd.read_pickle(pickle_path)
    except FileNotFoundError:
        pass

    prompt_path = os.path.join('prompts', f'{strategy}.txt')
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    client = AzureOpenAIClient()
    ranker = Ranker(prompt_template=prompt_template, client=client)

    for idx, row in df.iterrows():
        query = row['query_x']
        product_lhs = product_row_to_dict(row[['product_name_x', 'product_description_x', 'product_class_x',
                                               'product_id_x', 'category hierarchy_x', 'grade_x']])
        product_rhs = product_row_to_dict(row[['product_name_y', 'product_description_y', 'product_class_y',
                                               'product_id_y', 'category hierarchy_y', 'grade_y']])
        if has_been_labeled(results_df, query, product_lhs, product_rhs):
            continue
        ground_truth = compare_products(product_lhs, product_rhs)
        prediction = ranker.rank(query, product_lhs, product_rhs) or 'Neither'
        results_df = pd.concat([results_df, pd.DataFrame([output_row(query, product_lhs, product_rhs,
                                                                     ground_truth, prediction)])],
                               ignore_index=True)
        try:
            results_df.to_pickle(pickle_path)
            results_df.to_csv(csv_path, index=False)
        except Exception:
            pass
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
        sample_size=150,
        test_size=100
    )
