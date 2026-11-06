
import pandas as pd
from src.shipment_prep import clean_shipments

# Load raw dataset
raw = pd.read_csv("data/raw_shipments.csv")

# Call cleaning function
clean_df, medians = clean_shipments(raw)

# Check results
print("Original Rows:", len(raw))
print("Cleaned Rows:", len(clean_df))

print("\nCalculated Medians:")
print(medians)

print("\nMissing Values:")
print(clean_df.isnull().sum())

print("\nService Types:")
print(clean_df["service_type"].value_counts())

print("\nCleaned Data Preview:")
print(clean_df.head())
