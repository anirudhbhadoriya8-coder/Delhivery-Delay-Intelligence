import pandas as pd

hubs = pd.read_csv("data/hubs.csv")
raw = pd.read_csv("data/raw_shipments.csv")
new = pd.read_csv("data/new_shipments.csv")

print("Hubs Shape:", hubs.shape)
print("Raw Shipments Shape:", raw.shape)
print("New Shipments Shape:", new.shape)

print("\nRaw Data Preview:")
print(raw.head())

print("\nMissing Values:")
print(raw.isnull().sum())