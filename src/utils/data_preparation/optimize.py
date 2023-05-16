import pandas as pd
import numpy as np
from typing import List

from .constants import THRESHOLD


def optimize_floats(df: pd.DataFrame) -> None:
    """
    Optimizes data space usage by casting float columns to smallest possible size

    :param df: DataFrame holding data
    """
    cols = df.select_dtypes(include=np.float_).columns.tolist()
    df[cols] = df[cols].apply(pd.to_numeric, downcast="float")


def optimize_ints(df: pd.DataFrame) -> None:
    """
    Optimizes data space usage by casting integer columns to smallest possible size

    :param df: DataFrame holding data
    """
    cols = df.select_dtypes(include=np.integer).columns
    min_vals = df[cols].min(axis=0)
    unsigned_idxs = np.where(min_vals >= 0)[0]
    signed_idxs = [i for i in range(len(cols)) if i not in unsigned_idxs]
    df[cols[unsigned_idxs]] = df[cols[unsigned_idxs]].apply(
        pd.to_numeric, downcast="unsigned"
    )
    df[cols[signed_idxs]] = df[cols[signed_idxs]].apply(
        pd.to_numeric, downcast="signed"
    )


def optimize_objects(
    df: pd.DataFrame, datetime_features: List[str], threshold: int = THRESHOLD
) -> None:
    """
    Optimizes data space usage by casting object columns to pd.category if less or equal to threshold % of entries are unique,
    and datetime_features to pd.datetime

    :param df: DataFrame holding data
    :param datetime_features: List of columns that can be casted to datetime, which significantly reduces space usage
    :param threshold: int from 0 to 100
    """
    for col in df.select_dtypes(include=np.object_):
        if col not in datetime_features:
            if not (type(df[col][0]) == list):
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values <= threshold / 100.0:
                    df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col])


def convert_to_hhmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts every column of df into a hhmm string format

    :returns: modified df
    """
    df = df.fillna(0).astype(np.int16).astype(str)
    for col in df.columns:
        df[col] = df[col].str.zfill(4)
        bad_idxs = df[col] >= "2400"
        df.loc[bad_idxs, col] = (
            (df.loc[bad_idxs, col].astype(np.int16) - 2400).astype(str).str.zfill(4)
        )

    return df


def optimize(
    df: pd.DataFrame, datetime_features: List[str] = [], flights_data: bool = False
) -> None:
    """
    Optimizes data space usage

    :param df: DataFrame holding data
    :param datetime_features: List of columns that can be casted to datetime, which significantly reduces space usage
    :param flights_data: special flag that triggers additional data conversions only for flights data
    """

    optimize_ints(df)
    optimize_floats(df)
    optimize_objects(df, datetime_features)

    if flights_data:
        dates = {
            "DepTime": "Departure",
            "CRSDepTime": "CRSDeparture",
            "ArrTime": "Arrival",
            "CRSArrTime": "CRSArrival",
        }

        df.loc[:, list(dates.keys())] = convert_to_hhmm(df.loc[:, list(dates.keys())])

        for original_name, new_name in dates.items():
            df[new_name] = pd.to_datetime(
                dict(
                    year=df["Year"],
                    month=df["Month"],
                    day=df["DayofMonth"],
                    hour=df[original_name].str[:2],
                    minute=df[original_name].str[2:],
                )
            )

        df.drop(
            columns=["Year", "Month", "DayofMonth"] + list(dates.keys()),
            inplace=True,
        )


def concatenate(dfs: List[pd.DataFrame], threshold: int = THRESHOLD) -> pd.DataFrame:
    """
    Concatenate while preserving categorical columns.

    :param dfs: list of DataFrames to concatenate
    :param threshold: target column will be left as categorical if unique values are less threshold % of all values
    """
    assert len(dfs) >= 1, "dfs cannot be empty"
    target_size = sum([df.shape[0] for df in dfs])

    for col in dfs[-1].select_dtypes(include="category").columns:
        # if not category than it must have been all empty
        uc = pd.api.types.union_categoricals(
            [df[col] for df in dfs if df[col].dtype == "category"]
        )
        if len(uc.categories) / target_size <= threshold / 100.0:
            for df in dfs:
                df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    return pd.concat(dfs, ignore_index=True)
