import pandas as pd

from shipment_prep import clean_shipments, build_features, FEATURES


# 1. Load datasets
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# 2. Clean shipment data
clean, medians = clean_shipments(raw)


# 3. Build features
df = build_features(clean, hubs)


# 4. Create delay target
df["is_delayed"] = (
    df["delivery_hours"] > df["promised_hours"]
).astype(int)


# 5. Check data quality
print("Raw rows:", len(raw))
print("Cleaned rows:", len(clean))
print("Rows after hub merge:", len(df))
print("Missing values in features and target:",
      df[FEATURES + ["is_delayed"]].isnull().sum().sum())

print("\nDelay count:")
print(df["is_delayed"].value_counts())

print("\nDelay rate:")
print(round(df["is_delayed"].mean(), 4))


# 6. Save modelling table
df.to_csv("outputs/modelling_table.csv", index=False)

print("\nModelling table saved successfully!")
# 7. Delay analysis by service type
service_delay = df.groupby("service_type").agg(
    total_shipments=("is_delayed", "count"),
    delayed_shipments=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean")
).reset_index()

service_delay["delay_rate"] = service_delay["delay_rate"].round(4)

print("\nDelay by Service Type:")
print(service_delay)


# 8. Delay analysis by hub tier
tier_delay = df.groupby("hub_tier").agg(
    total_shipments=("is_delayed", "count"),
    delayed_shipments=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean")
).reset_index()

tier_delay["delay_rate"] = tier_delay["delay_rate"].round(4)

print("\nDelay by Hub Tier:")
print(tier_delay)
# 9. Create distance bands
df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, float("inf")],
    labels=["0-200", "201-400", "401-600", "601+"]
)

# 10. Delay analysis by distance
distance_delay = df.groupby(
    "distance_band", observed=False
).agg(
    total_shipments=("is_delayed", "count"),
    delayed_shipments=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean")
).reset_index()

distance_delay["delay_rate"] = distance_delay["delay_rate"].round(4)

print("\nDelay by Distance Band:")
print(distance_delay)

# 11. Correlation analysis
correlation_columns = [
    "delivery_hours",
    "promised_hours",
    "distance_km",
    "is_delayed"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Save correlation matrix
correlation_matrix.to_csv(
    "outputs/correlation_matrix.csv"
)

print("\nCorrelation matrix saved successfully!")

# Save service delay analysis
service_delay.to_csv(
    "outputs/service_delay.csv", index=False
)

# Save hub tier delay analysis
tier_delay.to_csv(
    "outputs/tier_delay.csv", index=False
)

# Save distance delay analysis
distance_delay.to_csv(
    "outputs/distance_delay.csv", index=False
)

# 12. Delivery hours outlier analysis using IQR

Q1 = df["delivery_hours"].quantile(0.25)
Q3 = df["delivery_hours"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["delivery_hours"] < lower_bound) |
    (df["delivery_hours"] > upper_bound)
]

print("\nOutlier Analysis:")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower Bound:", round(lower_bound, 2))
print("Upper Bound:", round(upper_bound, 2))
print("Outlier Count:", len(outliers))