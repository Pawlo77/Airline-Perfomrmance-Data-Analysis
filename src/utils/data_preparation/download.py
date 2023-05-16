import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webelement import WebElement

from .constants import DATASETS_FOLDER, ALLOWED_DELAY, TIME_DELTA, URL


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


def download(dir: str = DATASETS_FOLDER) -> None:
    """
    Downloads data into a dir directory

    .. warning:: This function strongly relies on the URL structure. Any errors are most likely caused by its chenges.

    :param dir: target data directory
    """
    assert os.path.exists(dir), "Directory does not exist"

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
