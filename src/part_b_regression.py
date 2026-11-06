import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shipment_prep import FEATURES


# 1. Load modelling dataset
df = pd.read_csv("outputs/modelling_table.csv")

# 2. Separate features and target
X = df[FEATURES]
y = df["delivery_hours"]

# 3. First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# 4. Second split: temporary into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

# 5. Display split sizes
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# 6. StandardScaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nScaled shapes:")
print("Train:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)

print("\nMaximum absolute training mean:",
      abs(X_train_scaled.mean(axis=0)).max())

print("Maximum absolute test mean:",
      abs(X_test_scaled.mean(axis=0)).max())

from sklearn.metrics import mean_squared_error
import numpy as np

# 7. Baseline model
baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_prediction)
)

print("\nBaseline Model:")
print("Training mean:", round(y_train.mean(), 2))
print("Validation RMSE:", round(baseline_rmse, 2))

from sklearn.tree import DecisionTreeRegressor

# 8. Decision Tree depth tuning
depths = [1, 2, 3, 4, 6, 8, 12, None]

print("\nDecision Tree Results:")

tree_results = []

for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_pred)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_pred)
    )

    tree_results.append({
        "depth": depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

    print(
        f"Depth: {depth}, "
        f"Train RMSE: {train_rmse:.2f}, "
        f"Validation RMSE: {val_rmse:.2f}"
    )

# Select best depth: lowest validation RMSE
best_tree_result = min(
    tree_results,
    key=lambda x: (x["val_rmse"], float("inf") if x["depth"] is None else x["depth"])
)

print("\nBest Tree Depth:", best_tree_result["depth"])
print("Best Validation RMSE:", round(best_tree_result["val_rmse"], 2))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline


# 9. Define regression models

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=best_tree_result["depth"],
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# 10. Train and evaluate models

print("\nRegression Model Comparison:")

for name, model in models.items():

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_pred)
    )

    print(f"{name}: Validation RMSE = {val_rmse:.2f}")

    from sklearn.metrics import mean_absolute_error, r2_score

# 11. Complete model evaluation

print("\nComplete Model Evaluation:")

for name, model in models.items():

    val_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, val_pred)
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, val_pred)

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 3))

    # 12. Gradient Boosting Learning Rate Experiment

learning_rates = [0.01, 0.1, 1.0]

print("\nGradient Boosting Learning Rate Experiment:")

for rate in learning_rates:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    val_pred = gb_model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)

    print(
        f"Learning Rate: {rate}, "
        f"RMSE: {rmse:.2f}, "
        f"R2: {r2:.3f}"
    )

    from sklearn.model_selection import KFold, cross_validate

# 13. Top 2 Models - 5 Fold Cross Validation

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

top_models = {
    "Linear Regression": models["Linear Regression"],
    "Gradient Boosting": models["Gradient Boosting"]
}

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n5-Fold Cross Validation Results:")

for name, model in top_models.items():

    scores = cross_validate(
        model,
        X_train_val,
        y_train_val,
        cv=kfold,
        scoring="r2"
    )

    print(f"\n{name}")
    print("Fold R2 Scores:", scores["test_score"].round(3))
    print("Mean R2:", round(scores["test_score"].mean(), 3))
    print("Standard Deviation:", round(scores["test_score"].std(), 3))

 # 14. Select Final Model Using Validation RMSE

best_model_name = "Linear Regression"
final_model = models[best_model_name]

print("\nSelected Final Model:", best_model_name)

# 15. Final Test Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train final model on combined training + validation data
final_model.fit(X_train_val, y_train_val)

# Predict on untouched test data
test_predictions = final_model.predict(X_test)

# Calculate evaluation metrics
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("\nFinal Test Evaluation:")
print("Selected Model:", best_model_name)
print("Test MAE:", round(test_mae, 2))
print("Test MSE:", round(test_mse, 2))
print("Test RMSE:", round(test_rmse, 2))
print("Test R2:", round(test_r2, 3))