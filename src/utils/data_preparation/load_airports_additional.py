import pandas as pd
import os
from .constants import DATASETS_FOLDER

URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"


def load_airports_details():
    try:
        airports_details = pd.read_pickle(
            os.path.join(DATASETS_FOLDER, "airports_details.pkl")
        )
    except:
        airports_details = pd.read_csv(
            URL,
            header=None,
            names=[
                "airportID",
                "name",
                "city",
                "country",
                "iata",
                "icao",
                "lat",
                "lon",
                "altitude",
                "timezone",
                "dst",
                "tz",
                "type",
                "source",
            ],
        )
        airports_details.to_pickle(
            os.path.join(DATASETS_FOLDER, "airports_details.pkl")
        )
    return airports_details
