import json
import math
from typing import List, Tuple

import plotly.graph_objects as go

import pandas as pd

from definitions import LATITUDES_BY_AIRPORT, LONGITUDES_BY_AIRPORT, POPULATION_DATA_DF, \
    INDEX, AIRPORTS, MISSING_DATA_BY_CITY, ORG, LOGGER, DATA_DIR, QUERY_API, \
    AUGMENTED_TEMPERATURE_DATA_PATH, MISSING_DATA_BY_CITY_PATH, MODE, Mode, TOTAL_POPULATION, VISITED, \
    DAILY_SEASONAL_WEIGHTED_TEMPERATURE_GRAPH_PATH, MONTHLY_WEIGHTED_TEMPERATURE_GRAPH_PATH, MISSING_DATA_GRAPH_PATH

pd.options.plotting.backend = "plotly"


def get_nearest_city_from_airport(airport: str) -> Tuple[str, str]:
    """
    Gets the nearest city in POPULATION_DATA_DF from an airport
    :param airport: Airport string code, see station_code in temperature_data.csv.
    :return: Tuple of (city, state)
    """
    latitude = LATITUDES_BY_AIRPORT[airport]
    longitude = LONGITUDES_BY_AIRPORT[airport]
    min_distance = math.inf
    city = state = None
    for i, row in POPULATION_DATA_DF.iterrows():
        current_distance = math.dist((latitude, longitude), (row['Lat'], row['Lon']))
        if current_distance < min_distance:
            min_distance = current_distance
            city, state = row['City'], row['State']

    return city, state


def knn(city: str, state: str, k: int = 3, radius: int = .22) -> List[Tuple[str, str]]:
    """
    Gets the nearest k cities in POPULATION_DATA_DF from a city within the bounds of radius. Skips any cities in
    VISITED so that they are not double counted.
    :param city: City to perform knn on
    :param state: State to perform knn on
    :param k: Number of neighboring cities to find
    :param radius: Radius to search
    :return: List of (city, state) tuples
    """
    latitude = POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == city) &
                                      (POPULATION_DATA_DF['State'] == state)]['Lat'].values[0]
    longitude = POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == city) &
                                       (POPULATION_DATA_DF['State'] == state)]['Lon'].values[0]
    neighbors = []
    for i, row in POPULATION_DATA_DF.iterrows():
        if (row['City'], row['State']) in VISITED:
            continue
        current_distance = math.dist((latitude, longitude), (row['Lat'], row['Lon']))
        if current_distance < radius:
            city, state = row['City'], row['State']
            neighbors.append((city, state, current_distance))
            VISITED.add((city, state))

    if len(neighbors) > 1:
        neighbors = sorted(neighbors, key=lambda x: x[2])[1:k+1]
    return [(neighbor[0], neighbor[1]) for neighbor in neighbors]


def augment_temperature_data() -> pd.DataFrame:
    if MODE == Mode.WRITE:
        dfs = []
        for airport in AIRPORTS:
            LOGGER.info(f'Augmenting {airport}')
            city, state = get_nearest_city_from_airport(airport)
            for field in ['temp_max_c', 'temp_mean_c', 'temp_min_c']:
                query = f"""from(bucket: "Population Weighted Daily Temperature")
                            |> range(start: 2015-01-01T00:00:00Z)
                            |> filter(fn: (r) =>
                                r._measurement == "Ticker" and
                                r.station_code == "{airport}" and
                                r._field == "{field}")"""
                df = QUERY_API.query_data_frame(query, org=ORG).drop(
                    columns=['result', '_start', '_stop', 'table', '_measurement'])
                df.set_index('_time', inplace=True)
                df = df.reindex(INDEX)
                MISSING_DATA_BY_CITY[city] = [str(index) for index, row in df.iterrows() if row.isnull().any()]
                df['station_code'].fillna(airport, inplace=True)
                df['_field'].fillna(field, inplace=True)
                df['_value'].interpolate(method='time', inplace=True)
                dfs.append(df)

        all_df = pd.concat(dfs)
        all_df.to_csv(AUGMENTED_TEMPERATURE_DATA_PATH)
        with open(MISSING_DATA_BY_CITY_PATH, 'w', encoding='utf-8') as f:
            json.dump(MISSING_DATA_BY_CITY, f, ensure_ascii=False, indent=4)

        return all_df

    all_df = pd.read_csv(AUGMENTED_TEMPERATURE_DATA_PATH)
    all_df.set_index('_time', inplace=True)
    return all_df


def population_weighted_temperatures() -> Tuple[pd.Series]:
    population_weighted_daily_mean_temperature_ss = []
    population_weighted_daily_max_temperature_ss = []
    population_weighted_daily_min_temperature_ss = []
    sum_populations = 0

    for airport in AIRPORTS:
        city, state = get_nearest_city_from_airport(airport)
        population = POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == city) &
                                            (POPULATION_DATA_DF['State'] == state)]['population'].values[0]
        sum_populations += population
        VISITED.add((city, state))

    for airport in AIRPORTS:
        city, state = get_nearest_city_from_airport(airport)
        for neighbor_city, neighbor_state in knn(city, state):
            population = POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == neighbor_city) &
                                                (POPULATION_DATA_DF['State'] == neighbor_state)]['population'].values[0]
            sum_populations += population

    for airport in AIRPORTS:
        city, state = get_nearest_city_from_airport(airport)
        mean_temperature = TEMPERATURE_DATA_DF.loc[(TEMPERATURE_DATA_DF['station_code'] == airport) &
                                                   (TEMPERATURE_DATA_DF['_field'] == 'temp_mean_c')]
        max_temperature = TEMPERATURE_DATA_DF.loc[(TEMPERATURE_DATA_DF['station_code'] == airport) &
                                                   (TEMPERATURE_DATA_DF['_field'] == 'temp_max_c')]
        min_temperature = TEMPERATURE_DATA_DF.loc[(TEMPERATURE_DATA_DF['station_code'] == airport) &
                                                   (TEMPERATURE_DATA_DF['_field'] == 'temp_min_c')]
        population = POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == city) &
                                            (POPULATION_DATA_DF['State'] == state)]['population'].values[0]
        for neighbor_city, neighbor_state in knn(city, state):
            population += POPULATION_DATA_DF.loc[(POPULATION_DATA_DF['City'] == neighbor_city) &
                                                 (POPULATION_DATA_DF['State'] == neighbor_state)]['population'].values[0]
        coefficient = population / sum_populations
        population_weighted_daily_mean_temperature_ss.append(mean_temperature['_value'].multiply(coefficient))
        population_weighted_daily_max_temperature_ss.append(max_temperature['_value'].multiply(coefficient))
        population_weighted_daily_min_temperature_ss.append(min_temperature['_value'].multiply(coefficient))

    LOGGER.info(f'Ratio of US population used: {sum_populations/TOTAL_POPULATION}')

    return sum(population_weighted_daily_mean_temperature_ss), sum(population_weighted_daily_max_temperature_ss),\
           sum(population_weighted_daily_min_temperature_ss)


def main():
    population_weighted_mean_temperatures_ts, population_weighted_max_temperatures_ts, \
        population_weighted_min_temperatures_ts = population_weighted_temperatures()
    population_weighted_mean_temperatures_ts.index = population_weighted_mean_temperatures_ts.index.astype(
        'datetime64[ns, UTC]')
    population_weighted_max_temperatures_ts.index = population_weighted_max_temperatures_ts.index.astype(
        'datetime64[ns, UTC]')
    population_weighted_min_temperatures_ts.index = population_weighted_min_temperatures_ts.index.astype(
        'datetime64[ns, UTC]')
    seasonal_mean = population_weighted_mean_temperatures_ts.resample('3M').mean()
    seasonal_max = population_weighted_max_temperatures_ts.resample('3M').max()
    seasonal_min = population_weighted_min_temperatures_ts.resample('3M').min()

    data = [go.Scatter(x=population_weighted_mean_temperatures_ts.index,
                       y=population_weighted_mean_temperatures_ts.values, name='Temperature'),
            go.Scatter(x=seasonal_mean.index,
                       y=seasonal_mean.values, name='Seasonal Average'),
            go.Scatter(x=seasonal_max.index,
                       y=seasonal_max.values, name='Seasonal Max'),
            go.Scatter(x=seasonal_min.index,
                       y=seasonal_min.values, name='Seasonal Min')]
    fig = go.Figure(data=data)
    fig.show()
    fig.write_html(DAILY_SEASONAL_WEIGHTED_TEMPERATURE_GRAPH_PATH)

    p_22 = pd.concat([population_weighted_mean_temperatures_ts.resample('1M').mean(),
                      population_weighted_max_temperatures_ts.resample('1M').max(),
                      population_weighted_min_temperatures_ts.resample('1M').min()], axis=1)

    p_22.columns = ['Monthly Average', 'Monthly Min', 'Monthly Max']

    fig = p_22.plot(title="US Population Weighted Daily Temperature Monthly")
    fig.show()
    fig.write_html(MONTHLY_WEIGHTED_TEMPERATURE_GRAPH_PATH)

    d = pd.DataFrame(0, index=INDEX, columns=MISSING_DATA_BY_CITY.keys())
    for k, v in MISSING_DATA_BY_CITY.items():
        for j in v:
            d[k][j] = 1

    fig = d.plot(title="Data Missing/Projected")
    fig.show()
    fig.write_html(MISSING_DATA_GRAPH_PATH)


if __name__ == '__main__':
    TEMPERATURE_DATA_DF = augment_temperature_data()
    main()
