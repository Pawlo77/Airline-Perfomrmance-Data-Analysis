import os
import warnings
import time
import pandas as pd

from zipfile import ZipFile

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webelement import WebElement


URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7#"
ROOT_DIR = os.path.split(os.path.split(__file__)[0])[0]
DATASETS_FOLDER = os.path.join(ROOT_DIR, "datasets")
ALLOWED_DELAY = 3
TIME_DELTA = 0.25


class CustomTimeoutException(Exception):
    """Custom timeout exception for Selenium WebDriver"""

    def __init__(self, message="Unable to download the data"):
        super().__init__(message)


def get_element(
    driver: webdriver.Chrome, xpath: str, timeout: int = ALLOWED_DELAY
) -> WebElement:
    """
    Waits up to timeout seconds for the element pointed by xpath to to appear on site.

    :param driver: a driver with loaded page
    :param xpath: an element's xpath on page
    :param timeout: the number of seconds to wait unitl raising an error
    :raises: CustomTimeoutException
    :returns: Desired Element
    """
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    xpath,
                )
            )
        )
    except TimeoutException:
        raise CustomTimeoutException
    else:
        return el


def downloaded(dir: str = DATASETS_FOLDER) -> bool:
    """
    Waits until file is downloaded. It has to be the only one in the dir

    :param dir: the directory into which some file is being downloaded
    """
    return os.listdir(dir)[0].split(".")[-1] != "crdownload"


def prepare_data(dir: str = DATASETS_FOLDER) -> None:
    """
    Downloads and extracts data. It assumes 3 possible situations:
    - dir is empty, so it downloads and extracts data on its own,
    - dir contains only a zip archive, so it extracts it on its own,
    - dir contains both a zip archive and its extracted data, it does nothing.

    .. warning:: This function strongly relies on the URL structure. Any errors are most likely caused by its chenges.

    :param dir: target data directory
    """

    # if data is not downloaded onto local machine
    if not os.path.exists(dir):
        warnings.warn("Dataset not found, downloading now...")
        os.makedirs(dir)

        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--headless=new")
        prefs = {"download.default_directory": dir}
        options.add_experimental_option("prefs", prefs)

        with webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        ) as driver:
            driver.get(URL)

            accessBtn = get_element(
                driver,
                "//button[@class='btn btn-primary btn-access-dataset dropdown-toggle']",
            )
            accessBtn.click()
            downloadBtn = get_element(
                driver, "//a[@class='ui-commandlink ui-widget btn-download']"
            )
            downloadBtn.click()

            # wait for download to start and finish
            # if waiting for start exceeds timeout raise exception
            # if for last 4 TIME_DELTA seconds file_size is the same, raise exception
            waiting_to_start = 0
            last_sizes = []
            not_started = lambda: len(os.listdir(dir)) == 0
            while not_started() or not downloaded():
                time.sleep(TIME_DELTA)
                if not_started():
                    waiting_to_start += TIME_DELTA
                    # waiting for download to start takes usually longer
                    if waiting_to_start > 3 * ALLOWED_DELAY:
                        raise CustomTimeoutException
                else:
                    file = os.path.join(dir, os.listdir(dir)[0])
                    file_size = os.path.getsize(file)
                    if len(last_sizes) < 4:
                        last_sizes.append(file_size)
                    else:
                        if last_sizes[0] == file_size:
                            raise CustomTimeoutException
                        last_sizes = last_sizes[1:] + [file_size]

    # if data is still only a zip archive
    if len(os.listdir(dir)) == 1:
        zip_file = os.path.join(dir, os.listdir(dir)[0])
        with ZipFile(zip_file, "r") as f:
            f.extractall(dir)

    # decompress the bz2 archives if they aren't already decompressed
    # remove them to save storage
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".bz2"):
            filepath = os.path.join(dir, filename)
            newfilepath = os.path.splitext(filepath)[0]
            newfilepath = os.path.splitext(newfilepath)[0] + ".pkl"
            if not os.path.exists(newfilepath):
                df = pd.read_csv(filepath, compression="bz2", encoding="ISO-8859-1")
                # df.to_pickle(newfilepath, compression="bz2")
                df.to_pickle(newfilepath)
                os.remove(filepath)


def load_flights(
    years: str | list = "all", cols: list = None, dir: str = DATASETS_FOLDER
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

    return pd.concat(flights, ignore_index=True)


def load_csv(filename: str, dir: str = DATASETS_FOLDER):
    """
    Utility function that loads a CSV file into a pd.DataFrame

    :param str filename: file name
    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return pd.read_csv(os.path.join(dir, filename))


def load_airports(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads airports from airports.csv

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_csv("airports.csv", dir)


def load_carriers(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads carriers data from carriers.csv

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_csv("carriers.csv", dir)


def load_plane_data(dir: str = DATASETS_FOLDER) -> pd.DataFrame:
    """
    Loads plane data from plane-data.csv

    :param dir: target data directory
    :returns: DataFrame with loaded data
    """
    return load_csv("plane-data.csv", dir)


if __name__ == "__main__":
    prepare_data()
