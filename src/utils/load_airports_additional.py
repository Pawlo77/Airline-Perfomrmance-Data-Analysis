import pandas as pd

url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'

def load_airports_details():
    return pd.read_csv(url, header=None, names=['airportID', 'name', 'city', 'country', 'iata', 'icao', 'lat', 'lon', 'altitude', 'timezone', 'dst', 'tz', 'type', 'source'])

def main():
    load_airports_details()

if __name__ == "__main__":
    main()