import numpy as np
import pandas as pd

SAMPLE_POOL_SIZE = 10_000


def _wands_data_merged():
    """Load WANDS tables and merge into a single labeled dataframe."""
    try:
        products = pd.read_csv('data/WANDS/dataset/product.csv', delimiter='\t')
        queries = pd.read_csv('data/WANDS/dataset/query.csv', delimiter='\t')
        labels = pd.read_csv('data/WANDS/dataset/label.csv', delimiter='\t')
    except FileNotFoundError:
        msg = ("Please download the WANDS dataset from https://github.com/wayfair/WANDS/" +
               "and place it in the data folder")
        raise FileNotFoundError(msg)
    labels.loc[labels['label'] == 'Exact', 'grade'] = 2
    labels.loc[labels['label'] == 'Partial', 'grade'] = 1
    labels.loc[labels['label'] == 'Irrelevant', 'grade'] = 0
    labels = labels.merge(queries, how='left', on='query_id')
    labels = labels.merge(products, how='left', on='product_id')
    return labels


def pairwise_df(sample_size, seed=42):
    """
    Build a pairwise dataset of product comparisons for the same query.
    sample_size controls the final number of rows returned.
    """
    labels = _wands_data_merged()

    # Sample a pool to make the pairwise join tractable.
    labels = labels.sample(SAMPLE_POOL_SIZE, random_state=seed)

    # Get pairwise
    pairwise = labels.merge(labels, on='query_id')
    # Shuffle completely, otherwise they're somewhat sorted on query
    pairwise = pairwise.sample(frac=1, random_state=seed)

    # Drop same id
    pairwise = pairwise[pairwise['product_id_x'] != pairwise['product_id_y']]

    # Drop same rating
    pairwise = pairwise[pairwise['label_x'] != pairwise['label_y']]

    assert sample_size <= len(pairwise), f"Only {len(pairwise)} rows available"
    return pairwise.head(sample_size)


def train_test_split(
        df: pd.DataFrame,
        test_size: int,
        seed: int = 42
):
    """Randomly split a dataframe into train and test subsets."""
    rng = np.random.default_rng(seed)
    assert 0 < test_size < len(df), "test_size must be between 1 and len(df)-1"

    idx = rng.choice(len(df), size=test_size, replace=False)
    test_df = df.iloc[idx].reset_index(drop=True)
    train_df = df.drop(df.index[idx]).reset_index(drop=True)

    return train_df, test_df


def pairwise_split(
        sample_size: int,
        test_size: int,
        seed: int = 42
):
    """Convenience wrapper: build pairwise data and split into train/test."""
    pairwise = pairwise_df(sample_size, seed)
    train_df, test_df = train_test_split(pairwise, test_size=test_size, seed=seed)
    return train_df, test_df
