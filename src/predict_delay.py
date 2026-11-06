
import pandas as pd
import joblib
from shipment_prep import clean_shipments, build_features

# 1. Load input files
new_shipments = pd.read_csv("data/new_shipments.csv")
original = new_shipments.copy()

hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")

threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_df = pd.read_csv("models/impute_values.csv")

medians = dict(
    zip(impute_df["column"], impute_df["median"])
)

print("New shipments loaded:", len(original))

# 2. Preserve original row order
raw_for_cleaning = new_shipments.copy()
raw_for_cleaning["_row_order"] = range(len(raw_for_cleaning))

# 3. Clean data
cleaned, _ = clean_shipments(
    raw_for_cleaning,
    medians=medians
)

cleaned = cleaned.sort_values("_row_order")

# 4. Build features
featured = build_features(cleaned, hubs)
featured = featured.sort_values("_row_order")

# 5. Arrange features in saved model input order
X_new = featured[feature_order]

# 6. Predict delay probability
probabilities = model.predict_proba(X_new)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

# 7. Attach hub segments
result = featured[["_row_order", "hub_id"]].merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
).sort_values("_row_order")

# 8. Assign recommended actions
def assign_action(prediction, segment):
    if prediction == 1:
        if segment == "Escalate":
            return "Priority intervention"
        elif segment == "Watch":
            return "Monitor and alert"
        else:
            return "Proactive customer alert"

    return "Normal monitoring"


actions = [
    assign_action(pred, segment)
    for pred, segment in zip(predictions, result["segment"])
]

# 9. Preserve original columns and add prediction results
final_output = original.copy()

final_output["delay_probability"] = probabilities
final_output["predicted_delay"] = predictions
final_output["segment"] = result["segment"].values
final_output["recommended_action"] = actions

# 10. Save output
final_output.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)

print("\nPrediction completed!")
print("Total shipments:", len(final_output))
print("Predicted delays:", int(final_output["predicted_delay"].sum()))
print("Output saved: outputs/new_shipment_actions.csv")

print("\nRecommended actions:")
print(final_output["recommended_action"].value_counts())
