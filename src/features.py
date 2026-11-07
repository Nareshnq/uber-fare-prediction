"""
Helper functions for the Uber fare prediction project.

I moved these out of the notebook once the cleaning and feature steps got
long enough that scrolling past them every time was annoying. The notebook
imports from here so the logic only lives in one place.
"""

import numpy as np
import pandas as pd

# Rough box around NYC (Staten Island to the Bronx, and out to JFK).
# The original dataset has points in the ocean and even in Antarctica,
# so anything outside this box gets dropped.
NYC_BOUNDS = {
    "lon_min": -74.3, "lon_max": -73.7,
    "lat_min": 40.5,  "lat_max": 41.0,
}

# Reference points I used for the location features
MIDTOWN = (40.7549, -73.9840)   # around Grand Central
AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
}
AIRPORT_RADIUS_KM = 2.0

COORD_COLS = ["pickup_longitude", "pickup_latitude",
              "dropoff_longitude", "dropoff_latitude"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on Earth, in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Direction of travel in degrees (-180 to 180). 0 = north, 90 = east."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y))


def clean_data(df, verbose=True):
    """
    Apply all the cleaning rules and print how many rows each one removed.
    Returns a new dataframe.
    """
    df = df.copy()
    start = len(df)
    log = []

    def drop(mask_to_keep, reason):
        nonlocal df
        removed = int((~mask_to_keep).sum())
        df = df[mask_to_keep]
        log.append((reason, removed))

    # leftover index column + the key column (it's just the timestamp again)
    df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")

    drop(df.notna().all(axis=1), "missing values")

    # NYC minimum fare was $2.50 during this period. Above $200 is
    # almost always a data entry problem in this dataset.
    drop(df["fare_amount"].between(2.5, 200), "fare outside $2.50 - $200")

    # 0 passengers or 208 passengers don't make sense for a car
    drop(df["passenger_count"].between(1, 6), "passenger count not 1-6")

    in_box = (
        df["pickup_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        & df["dropoff_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        & df["pickup_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        & df["dropoff_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
    )
    drop(in_box, "coordinates outside NYC")

    dist = haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                        df["dropoff_latitude"], df["dropoff_longitude"])

    # pickup == dropoff (under 50 m). Could be cancelled trips or GPS
    # glitches, either way the distance tells the model nothing.
    drop(dist > 0.05, "zero-distance trips")
    dist = dist[df.index]

    # under 1 km but over $50 - most likely bad coordinates
    drop(~((dist < 1) & (df["fare_amount"] > 50)), "short trip with huge fare")

    drop(~df.duplicated(), "duplicates")

    if verbose:
        for reason, n in log:
            print(f"  {reason:<28} {n:>6,} rows")
        print(f"  {'-' * 36}")
        print(f"  kept {len(df):,} of {start:,} rows "
              f"({(start - len(df)) / start:.1%} removed)")

    return df.reset_index(drop=True)


def add_features(df):
    """Add the time and location features used by the models."""
    df = df.copy()

    # The timestamps end in "UTC", but the hourly pattern (quietest at 5 AM,
    # busiest at 7 PM) only makes sense if they're already New York local
    # time. Converting them from UTC shifted everything 4-5 hours, so I just
    # drop the label and use them as-is. See section 4 of the notebook.
    ts = pd.to_datetime(df["pickup_datetime"].str.replace(" UTC", "", regex=False))
    df["pickup_datetime"] = ts
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday          # 0 = Monday
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = (
        (df["weekday"] < 5)
        & (df["hour"].between(7, 9) | df["hour"].between(16, 19))
    ).astype(int)
    df["is_night"] = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)

    plat, plon = df["pickup_latitude"], df["pickup_longitude"]
    dlat, dlon = df["dropoff_latitude"], df["dropoff_longitude"]

    df["distance_km"] = haversine_km(plat, plon, dlat, dlon)
    df["bearing"] = bearing_deg(plat, plon, dlat, dlon)
    df["pickup_to_midtown_km"] = haversine_km(plat, plon, *MIDTOWN)
    df["dropoff_to_midtown_km"] = haversine_km(dlat, dlon, *MIDTOWN)

    # Flag trips that start or end near an airport. JFK in particular
    # had a flat fare to Manhattan, so distance alone can't explain it.
    for name, (lat, lon) in AIRPORTS.items():
        near_pickup = haversine_km(plat, plon, lat, lon) < AIRPORT_RADIUS_KM
        near_dropoff = haversine_km(dlat, dlon, lat, lon) < AIRPORT_RADIUS_KM
        df[f"{name}_trip"] = (near_pickup | near_dropoff).astype(int)

    return df


# Feature lists, kept here so the notebook and any future script agree
BASIC_FEATURES = [
    "distance_km", "passenger_count", "hour", "weekday",
    "month", "year", "is_rush_hour",
]

ALL_FEATURES = BASIC_FEATURES + [
    "is_weekend", "is_night", "bearing",
    "pickup_to_midtown_km", "dropoff_to_midtown_km",
    "jfk_trip", "lga_trip", "ewr_trip",
] + COORD_COLS
