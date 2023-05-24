import os

import numpy as np


ROOT_DIR = os.path.split(
    os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
)[0]
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

MONTHS = np.array(
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
)

WEEK_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

REQUIRE = [
    "UniqueCarrier",
    "CRSElapsedTime",
    "DepDelay",
    "ArrDelay",
    "TailNum",
    "Cancelled",
    "CancellationCode",
    "DayOfWeek",
    "Origin",
    "Dest",
    "Departure",
    "Arrival",
]
