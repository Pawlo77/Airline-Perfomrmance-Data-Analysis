import os


THRESHOLD = 50

ROOT_DIR = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
DATASETS_FOLDER = os.path.join(ROOT_DIR, "datasets")

URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7#"
ALLOWED_DELAY = 3
TIME_DELTA = 0.25
