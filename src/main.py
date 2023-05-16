import os

from utils import generate_charts
from utils.charts.constants import PLOTS_DIR, ROOT_DIR
from utils.charts.generate_charts import main as charts_main


if __name__ == "__main__":
    if not os.path.exists(PLOTS_DIR):
        charts_main()
        generate_charts()
        generate_charts(["1988"])
        generate_charts(["2007"])
