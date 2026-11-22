import numpy as np
import pandas as pd

from src.preprocess.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


CATEGORY_VALUES = {
    "meal_package": ["Breakfast", "Full Board", "Half Board", "Room Only"],
    "room_category": [f"Type {letter}" for letter in "ABCDEFG"],
    "booking_channel": [
        "Airline", "Complimentary", "Corporate", "Offline Agent", "Online Travel Agent",
    ],
}


DTL_CATEGORICAL_COLUMNS = [
    column for column in CATEGORICAL_COLUMNS
    if column not in ("arrival_year", "arrival_day")
] + ["arrival_weekday"]


def prepare_dtl_features(df, indicators=True):
    X = df[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS].copy()
    X[NUMERICAL_COLUMNS] = X[NUMERICAL_COLUMNS].mask(X[NUMERICAL_COLUMNS] < 0)
    X["num_adults"] = X["num_adults"].mask(X["num_adults"] == 0)

    night_columns = ["weekday_nights", "weekend_nights"]
    no_nights = X[night_columns].eq(0).all(axis=1)
    X.loc[no_nights, night_columns] = np.nan

    date_columns = ["arrival_year", "arrival_month", "arrival_day"]
    date_parts = X[date_columns].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    invalid_date = date_parts.notna().all(axis=1) & pd.to_datetime(
        date_parts, errors="coerce"
    ).isna()
    X.loc[invalid_date, date_columns] = np.nan

    date_parts = X[date_columns].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    arrival = pd.to_datetime(date_parts, errors="coerce")
    booking = arrival - pd.to_timedelta(X["days_before_arrival"], unit="D")
    origin = pd.Timestamp("2023-01-01")
    X["arrival_date"] = (arrival - origin).dt.days
    X["booking_date"] = (booking - origin).dt.days
    X["arrival_weekday"] = arrival.dt.dayofweek
    X["total_nights"] = X["weekend_nights"] + X["weekday_nights"]
    X["total_price"] = X["nightly_rate_idr"] * X["total_nights"]

    for column in ["nightly_rate_idr", "special_requests", "num_children", "days_before_arrival"]:
        X[column + "_missing"] = X[column].isna()
    X = X.drop(columns=["arrival_day", "arrival_year"])

    if indicators:
        values_by_column = {
            "special_requests": [0, 1, 2],
            "num_adults": [1, 2],
            "num_children": [0],
            "weekday_nights": [1, 2, 3],
            "weekend_nights": [0, 1],
        }
        for column, values in values_by_column.items():
            for value in values:
                X[f"{column}={value}"] = (
                    X[column].eq(value).astype(float).where(X[column].notna())
                )
    for column, values in CATEGORY_VALUES.items():
        X[column] = X[column].map({value: i for i, value in enumerate(values)})
    return X
