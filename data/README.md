# Data

The dataset isn't stored in this repo. You can download it from Kaggle:

**[Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)**

Unzip it and put `uber.csv` in this folder:

```
data/
└── uber.csv
```

## Columns

| Column | Description |
|---|---|
| `key` | Unique ID for each trip (same as the pickup time) |
| `fare_amount` | Fare in USD (the **target**) |
| `pickup_datetime` | Pickup date and time (UTC) |
| `pickup_longitude`, `pickup_latitude` | Pickup location |
| `dropoff_longitude`, `dropoff_latitude` | Dropoff location |
| `passenger_count` | Number of passengers |

200,000 trips in New York City, 2009 to 2015.
