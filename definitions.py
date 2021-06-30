import json
from enum import Enum
from typing import List

import pandas as pd
import logging
import os

from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions


# Run Settings
class Mode(Enum):
    WRITE = 'WRITE'
    READ = 'READ'


MODE = Mode.READ

LOGGER = logging.getLogger(__name__)
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
TOKEN = "CHFsQ1Kaasm96F-mUJjIO-MBeJJTn9Tr80RTw5khaq8B5iOROryROOiM5Y2LQ6BkXwJ5yt3txFWR-2yPxDBx0w=="
ORG = "Foresight"
BUCKET = "Population Weighted Daily Temperature"
CLIENT = InfluxDBClient(url="https://us-west-2-1.aws.cloud2.influxdata.com", token=TOKEN, pool_size=50)
WRITE_API = CLIENT.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000))
QUERY_API = CLIENT.query_api()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
AIRPORT_DATA_PATH = os.path.join(DATA_DIR, 'airport_data.csv')
POPULATION_DATA_PATH = os.path.join(DATA_DIR, 'population_data.csv')
TEMPERATURE_DATA_PATH = os.path.join(DATA_DIR, 'temperature_data.csv')
AUGMENTED_TEMPERATURE_DATA_PATH = os.path.join(DATA_DIR, 'augmented_temperature_data.csv')
MISSING_DATA_BY_CITY_PATH = os.path.join(DATA_DIR, 'missing_data_by_city.json')
DAILY_SEASONAL_WEIGHTED_TEMPERATURE_GRAPH_PATH = os.path.join(DATA_DIR, 'daily_and_seasonal_weighted_temperature.html')
MONTHLY_WEIGHTED_TEMPERATURE_GRAPH_PATH = os.path.join(DATA_DIR, 'monthly_weighted_temperature.html')
MISSING_DATA_GRAPH_PATH = os.path.join(DATA_DIR, 'missing_data.html')

INDEX = pd.DatetimeIndex(pd.date_range('1/1/2015', '4/20/2021', freq='D', tz='utc'), name='_time')
AIRPORT_DATA_DF = pd.read_csv(AIRPORT_DATA_PATH)
TEMPERATURE_DATA_DF = pd.read_csv(TEMPERATURE_DATA_PATH)
TEMPERATURE_DATA_DF = TEMPERATURE_DATA_DF.set_index('location_date')

if MODE == Mode.WRITE:
    LOGGER.info(f"Clearing data...")
    CLIENT.delete_api().delete("2015-01-01T00:00:00Z", "2022-01-01T00:00:00Z", '_measurement="Ticker"', bucket=BUCKET,
                               org=ORG)
    LOGGER.info(f"Cleared data")

    LOGGER.info(f"Writing data...")
    WRITE_API.write(BUCKET, ORG, TEMPERATURE_DATA_DF, data_frame_measurement_name='Ticker',
                    data_frame_tag_columns=['station_code'])
    LOGGER.info(f"Data written")

POPULATION_DATA_DF = pd.read_csv(POPULATION_DATA_PATH)
TOTAL_POPULATION = POPULATION_DATA_DF['population'].sum()

AIRPORTS = TEMPERATURE_DATA_DF['station_code'].unique()
LATITUDES_BY_AIRPORT = {}
LONGITUDES_BY_AIRPORT = {}
_missing_airports = set()
for _airport in AIRPORTS:
    row = AIRPORT_DATA_DF[AIRPORT_DATA_DF.eq(_airport).any(1)]
    if not row.empty:
        lat, lon = row['coordinates'].values[0].split(', ')
        LATITUDES_BY_AIRPORT[_airport] = float(lat)
        LONGITUDES_BY_AIRPORT[_airport] = float(lon)
    else:
        _missing_airports.add(_airport)

if _missing_airports:
    LOGGER.warning(f'Missing airports: {_missing_airports}')


MISSING_DATA_BY_CITY = {}
if MODE == Mode.READ:
    with open(MISSING_DATA_BY_CITY_PATH) as f:
        MISSING_DATA_BY_CITY = json.load(f)

VISITED = set()
