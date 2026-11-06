import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean_df, medians = clean_shipments(raw)
feature_df = build_features(clean_df, hubs)

print("Rows after merge:", len(feature_df))
print("Feature count:", len(FEATURES))
print("Missing feature columns:", [col for col in FEATURES if col not in feature_df.columns])
print("Missing values in features:", feature_df[FEATURES].isnull().sum().sum())
print("Promised hours:", feature_df["promised_hours"].value_counts().to_dict())

print("\nFirst 5 rows:")
print(feature_df[FEATURES].head())