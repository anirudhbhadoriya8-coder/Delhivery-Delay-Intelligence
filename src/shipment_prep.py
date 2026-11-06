
import pandas as pd
import numpy as np

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]

def clean_shipments(raw, medians=None):

    # Step 1: Remove exact duplicate rows
    df = raw.drop_duplicates().copy()

    # Step 2: Clean service_type
    df["service_type"] = (
        df["service_type"]
        .str.strip()
        .str.title()
    )

    # Step 3: Fill missing payment_mode
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: Convert declared_value into numeric
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value_inr"],
        errors="coerce"
    )

    df = df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch_time into datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: Handle scanner-fault sentinel
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step 7: Calculate medians or use stored medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median())
        }

    # Step 8: Fill missing values
    for column, value in medians.items():
        df[column] = df[column].fillna(value)

    return df, medians
def build_features(clean, hubs):

    df = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left"
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df