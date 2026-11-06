
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from shipment_prep import FEATURES, clean_shipments, build_features


# ==========================================
# 1. CREATE OUTPUT FOLDERS
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA AND CONFIGURATION
# ==========================================

df = pd.read_csv("outputs/modelling_table.csv")
config = pd.read_csv("outputs/part_c_final_config.csv")

selected_features = config.loc[0, "selected_features"].split(",")
threshold = float(config.loc[0, "threshold"])

print("Selected features:", selected_features)
print("Threshold:", threshold)


# ==========================================
# 3. RECREATE PART C SPLIT
# ==========================================

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ==========================================
# 4. BUILD FINAL PIPELINE
# ==========================================

selector = SelectKBest(score_func=f_classif, k=7)

model = Pipeline([
    ("selector", selector),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])

model.fit(X_train, y_train)


# ==========================================
# 5. SAVE MODEL
# ==========================================

joblib.dump(model, "models/delay_model.pkl")

pd.DataFrame({
    "threshold": [threshold]
}).to_csv("models/delay_threshold.csv", index=False)

pd.DataFrame({"feature": FEATURES}).to_csv(
    "models/feature_order.csv", index=False
)
# ==========================================
# 6. SAVE IMPUTATION MEDIANS
# ==========================================

raw = pd.read_csv("data/raw_shipments.csv")

cleaned, medians = clean_shipments(raw)

pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)


# ==========================================
# 7. RELOAD MODEL CHECK
# ==========================================

reloaded_model = joblib.load("models/delay_model.pkl")

memory_probabilities = model.predict_proba(X_test)[:, 1]
reload_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_difference = abs(
    memory_probabilities - reload_probabilities
).max()

reload_check = pd.DataFrame({
    "test_rows_scored": [len(X_test)],
    "reload_max_abs_probability_diff": [max_difference],
    "threshold": [threshold]
})

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)

print("\nModel packaging completed!")
print("Test rows scored:", len(X_test))
print("Reload probability difference:", max_difference)
print("\nReload check:")
print(reload_check.to_string(index=False))
