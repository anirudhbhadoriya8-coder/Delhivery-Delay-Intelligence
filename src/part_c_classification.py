
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from shipment_prep import FEATURES, NOISE_COLUMNS


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("outputs/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 2. STRATIFIED TRAIN / VALIDATION / TEST
# ==========================================

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

print("\nSplit Sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nDelayed Class Counts:")
print("Train:", y_train.sum())
print("Validation:", y_val.sum())
print("Test:", y_test.sum())


# ==========================================
# 3. BLIND BASELINE
# ==========================================

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

print("\nBlind Baseline Results:")
print("Accuracy:", accuracy_score(y_val, baseline_predictions))
print("Precision:", precision_score(
    y_val, baseline_predictions, zero_division=0
))
print("Recall:", recall_score(
    y_val, baseline_predictions, zero_division=0
))
print("F1 Score:", f1_score(
    y_val, baseline_predictions, zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_val, baseline_predictions))


# ==========================================
# 4. TRAIN CLASSIFICATION MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print(f"\n{name}")
    print("Accuracy:", round(
        accuracy_score(y_val, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_val, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_val, predictions, zero_division=0), 4
    ))
    print("F1 Score:", round(
        f1_score(y_val, predictions, zero_division=0), 4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))


# ==========================================
# 5. FEATURE IMPORTANCE
# ==========================================

rf_model = models["Random Forest"]

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))


# ==========================================
# 6. NOISE FEATURE ANALYSIS
# ==========================================

noise_importance = importance_df[
    importance_df["Feature"].isin(NOISE_COLUMNS)
]

print("\nNoise Feature Importance:")
print(noise_importance.to_string(index=False))

print(
    "\nTotal Noise Importance:",
    round(noise_importance["Importance"].sum(), 4)
)


# ==========================================
# 7. SELECTKBEST FEATURE SELECTION
# ==========================================

print("\nSelectKBest Results:")

for k in [5, 7, 10, 12, 17]:

    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ])

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_val_selected)

    selected_features = [
        feature
        for feature, selected in zip(FEATURES, selector.get_support())
        if selected
    ]

    print(f"\nTop {k} Features:", selected_features)
    print("F1 Score:", round(f1_score(y_val, predictions), 4))


# ==========================================
# 8. SELECT BEST 7 FEATURES
# ==========================================

selector_7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

selected_features_final = [
    feature
    for feature, selected in zip(FEATURES, selector_7.get_support())
    if selected
]

print("\nSelected 7 Features:")
print(selected_features_final)


# ==========================================
# 9. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_7, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation F1:")
print(round(grid_search.best_score_, 4))

best_rf = grid_search.best_estimator_

rf_predictions = best_rf.predict(X_val_7)

print("\nTuned Random Forest Validation Results:")
print("F1 Score:", round(f1_score(y_val, rf_predictions), 4))
print("Recall:", round(recall_score(y_val, rf_predictions), 4))
print("Precision:", round(precision_score(y_val, rf_predictions), 4))


# ==========================================
# 10. THRESHOLD TUNING
# ==========================================

rf_probabilities = best_rf.predict_proba(X_val_7)[:, 1]

best_threshold = None
best_threshold_f1 = -1

print("\nThreshold Tuning Results:")

for threshold in [i / 100 for i in range(10, 91)]:

    predictions = (rf_probabilities >= threshold).astype(int)

    recall = recall_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions)

    if recall >= 0.80 and threshold > (best_threshold or 0):
        best_threshold = threshold
        best_threshold_f1 = f1

print("Selected Threshold:", best_threshold)
print("F1 Score:", round(best_threshold_f1, 4))


# ==========================================
# 11. FINAL VALIDATION RESULTS
# ==========================================

if best_threshold is None:
    raise ValueError("No threshold found with recall >= 0.80")

final_threshold_predictions = (
    rf_probabilities >= best_threshold
).astype(int)

print("\nFinal Threshold Results:")
print("Precision:", round(
    precision_score(y_val, final_threshold_predictions), 4
))
print("Recall:", round(
    recall_score(y_val, final_threshold_predictions), 4
))
print("Confusion Matrix:")
print(confusion_matrix(y_val, final_threshold_predictions))


# ==========================================
# 12. FINAL TEST EVALUATION
# ==========================================

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

X_train_val_7 = selector_7.transform(X_train_val)
X_test_7 = selector_7.transform(X_test)

final_rf = RandomForestClassifier(
    **grid_search.best_params_,
    random_state=42
)

final_rf.fit(X_train_val_7, y_train_val)

test_probabilities = final_rf.predict_proba(X_test_7)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFinal Test Evaluation:")
print("Accuracy:", round(
    accuracy_score(y_test, test_predictions), 4
))
print("Precision:", round(
    precision_score(y_test, test_predictions), 4
))
print("Recall:", round(
    recall_score(y_test, test_predictions), 4
))
print("F1 Score:", round(
    f1_score(y_test, test_predictions), 4
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# ==========================================
# 13. SAVE FINAL CONFIGURATION
# ==========================================

Path("outputs").mkdir(exist_ok=True)

config = pd.DataFrame({
    "model": ["Random Forest"],
    "selected_features": [
        ",".join(selected_features_final)
    ],
    "threshold": [best_threshold]
})

config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\nFinal configuration saved!")
print(config.to_string(index=False))
