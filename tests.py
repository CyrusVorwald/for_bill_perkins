import pandas as pd
import unittest

from definitions import TEMPERATURE_DATA_DF
from main import get_nearest_city_from_airport


class TestPart1(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_population_weighted_temperature(self) -> None:
        pass

    def test_get_nearest_city(self) -> None:
        airports = TEMPERATURE_DATA_DF['station_code'].unique()
        for airport in airports:
            print(airport, get_nearest_city_from_airport(airport))
