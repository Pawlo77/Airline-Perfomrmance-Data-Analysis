import logging

logging.basicConfig(level=logging.INFO)

from .data_preparation import (
    load_airports_details,
    load_flights,
    load_airports,
    load_carriers,
    load_plane_data
)

from .charts import generate_charts

load_flights = load_flights
load_airports = load_airports
load_carriers = load_carriers
load_plane_data = load_plane_data
generate_charts = generate_charts
