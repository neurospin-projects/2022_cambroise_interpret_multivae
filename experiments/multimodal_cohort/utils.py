import numpy as np
import pandas as pd


def extract_and_order_by(df, column_name, values):
    """
    """
    df = df[df[column_name].isin(values)]
    new_index = pd.Series(pd.Categorical(
        df[column_name], categories=values, ordered=True)
        ).sort_values().index
    df = df.reset_index(drop=True).loc[new_index]

    return df


def discretizer(values, method="auto"):
    """
    """
    bins = np.histogram_bin_edges(values, bins=method)
    new_values = np.digitize(values, bins=bins[1:], right=True)

    return new_values