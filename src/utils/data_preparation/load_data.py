import os
import sys
import logging
import pandas as pd
import traceback
import warnings

from typing import List
from zipfile import ZipFile
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from .download import download
from .optimize import optimize, concatenate
from .constants import DATASETS_FOLDER


def unpack(dir: str, filename: str, datetime_features: List[str] = []) -> None:
    """
    Unpacks a filename into a dir
    """
    warnings.simplefilter("ignore")

    try:
        compression = filename.endswith(".bz2")

        filepath = os.path.join(dir, filename)
        newfilepath = os.path.splitext(filepath)[0]
        if compression:
            newfilepath = os.path.splitext(newfilepath)[0] + ".pkl"
        else:
            newfilepath += ".pkl"

        if not os.path.exists(newfilepath):
            if compression:
                df = pd.read_csv(filepath, compression="bz2", encoding="ISO-8859-1")
            else:
                df = pd.read_csv(filepath, encoding="ISO-8859-1")

        # remove them to save storage, optimize their space usage
        os.remove(filepath)
        old_size = sys.getsizeof(df)
        optimize(df, datetime_features, flights_data=compression)
        new_size = sys.getsizeof(df)

        df.to_pickle(newfilepath)
        logging.info(
            f"Converted {filepath}. Original size {old_size} bytes shrinked to {new_size} bytes ({new_size/old_size:1.5f})"
        )
    except Exception as e:
        print(traceback.format_exc())
        raise e


def prepare_data(dir: str = DATASETS_FOLDER, datetime_features: List[str] = []) -> None:
    """
    Downloads and extracts data. It assumes 3 possible situations:
    - dir is empty, so it downloads and extracts data on its own,
    - dir contains only a zip archive, so it extracts it on its own,
    - dir contains both a zip archive and its extracted data, it does nothing.

    .. warning:: This function strongly relies on the URL structure. Any errors are most likely caused by its chenges.

    :param dir: target data directory
    :param datetime_features: List of columns that can be casted to datetime, which significantly reduces space usage
    """

    # if data is not downloaded onto local machine
    if not os.path.exists(dir):
        logging.warn("Dataset not found, it will take a while...")
        logging.info("Downloading data.")
        os.makedirs(dir)
        download(dir)

    # if data is still only a zip archive
    if len(os.listdir(dir)) == 1:
        logging.info("Extracting data.")
        zip_file = os.path.join(dir, os.listdir(dir)[0])
        with ZipFile(zip_file, "r") as f:
            f.extractall(dir)

    # decompress the bz2 archives if they aren't already decompressed
    # put them and all of .csv in .pkl format with optimised space usage
    args = [
        (dir, filename, datetime_features)
        for filename in sorted(os.listdir(dir))
        if filename.endswith(".bz2") or filename.endswith(".csv")
    ]
    # with ThreadPool() as p:
    with Pool() as p:
        p.starmap(unpack, args)


def load_flights(
    years: str | List[str] = "all", cols: List[str] = None, dir: str = DATASETS_FOLDER
) -> pd.DataFrame:
    """
    Loads flight data into memory

    :param years: "all" or all possible data, List of str from {"1987", ..., "2008"} for specific ones
    :param cols: desired columns to be loaded, if None entire data is loaded
    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    prepare_data(dir)
    assert len(years) > 0, "Must have at least one year specified"

    if years == "all":
        files = [
            os.path.join(dir, file)
            for file in sorted(os.listdir(dir))
            if file.endswith(".pkl") and file.split(".")[0].isnumeric()
        ]
    else:
        files = [
            os.path.join(dir, file)
            for file in sorted(os.listdir(dir))
            if file.split(".")[0] in years and file.endswith(".pkl")
        ]

    if cols is None:
        flights = [pd.read_pickle(file) for file in files]
    else:
        flights = [pd.read_pickle(file).loc[:, cols] for file in files]

    return concatenate(flights)


def load_pkl(filename: str, dir: str = DATASETS_FOLDER):
    """
    Utility function that loads a .pkl file into a pd.DataFrame

    :param str filename: file name
    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return pd.read_pickle(os.path.join(dir, filename))


def load_airports(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads airports from airports.csv

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_pkl("airports.pkl", dir)


def load_carriers(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads carriers data from carriers.pkl

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_pkl("carriers.pkl", dir)


def load_plane_data(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads plane data from plane-data.pkl

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_pkl("plane-data.pkl", dir)


if __name__ == "__main__":
    prepare_data()
