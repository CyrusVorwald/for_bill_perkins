Response to https://twitter.com/bp22/status/1409278019886370838

Run by cloning the project and entering the following in the terminal:
> pip install -r requirements.txt
> python3 main.py

I added airport code data (from here: https://github.com/datasets/airport-codes/blob/master/data/airport-codes.csv)
because the data sent is unclear. For example, KIAD is in VA, not DC; KDCA is in VA, not Washington.
Washington and Wash DC/Dulles both refer to Washington DC.

We are assuming the population of each city is a good weight to represent the average temperature in the US.
For this exercise, I loop over each airport, interpolate missing data, find the closest city to the airport,
take a simple coefficient of that city's population divided by the total population, and then do the same for k=3
neighboring cities within approximately 15 miles. This is pretty arbitrary and only represents about 40% of the total
US population. To account for more of the population, the temperature data would need to cover more land. With the
given dataset, there is a tradeoff between how much of the US population is represented, and accuracy of the representation.
Weather is highly variable and can differ greatly over a small area, so the further away the city is from a station,
the less accurate the representation will be.
In reality, population weighted daily temperature timeseries is calculated in a more regional way
(see https://www.eia.gov/outlooks/steo/special/pdf/2012_sp_04.pdf).

For some reason the InfluxDB Python client is extremely slow (the web client is extremely fast).
I created augmented_temperature_data.csv and missing_data_by_city.json to cache its use.
MODE can be switched to mode.WRITE if you want to go through the process of recreating everything.

1. Read CSV data
2. Interpolate missing data
3. Find closest city to each airport
4. Find k nearest neighboring city to that city, which have not already been visited
5. Create population weighted daily temperature timeseries
6. Write timeseries data to influx db cloud
7. Graph using plotly
8. Can view data visually in influx or by running main.py

To view data in influx, go to https://us-west-2-1.aws.cloud2.influxdata.com/orgs/7c5364d7744a8c09/data-explorer and
sign in.
Under 'From', select 'Population Weighted Daily Temperature'
Under '_measurement', select 'Ticker'

You can view any grouping of the following:
temperature,
seasonal_average,
monthly_average,
monthly_min,
monthly_max,
missing_data,
projected_data
