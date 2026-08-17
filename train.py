import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

df = pd.read_csv("data/netflix_titles.csv")

print("Original Shape:", df.shape)

print("\nMissing release years:")
print(df["release_year"].isnull().sum())


# Remove missing release years
df = df.dropna(subset=["release_year"])

# Convert release year to integer
df["release_year"] = df["release_year"].astype(int)


# ============================================================
# STEP 2 — ANALYZE HISTORICAL RELEASE PATTERNS
# ============================================================

# Create yearly content count
yearly_releases = (
    df.groupby("release_year")
    .size()
    .reset_index(name="content_count")
)

# Sort by year
yearly_releases = yearly_releases.sort_values("release_year")

print("\nYearly Release Data:")
print(yearly_releases.head(15))

print("\nLatest Years:")
print(yearly_releases.tail(10))


# Keep recent historical data
yearly_releases = yearly_releases[
    yearly_releases["release_year"] >= 2008
]

print("\nHistorical Data for Forecasting:")
print(yearly_releases)


# Save yearly data
yearly_releases.to_csv(
    "outputs/yearly_netflix_releases.csv",
    index=False
)

print("\nPrepared data saved successfully!")


# ============================================================
# STEP 3 — HISTORICAL TREND GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    yearly_releases["release_year"],
    yearly_releases["content_count"],
    marker="o"
)

plt.title("Netflix Content Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Releases")

plt.grid(True)
plt.show()


# ============================================================
# STEP 4 — MOVIES VS TV SHOWS ANALYSIS
# ============================================================

type_yearly = (
    df.groupby(["release_year", "type"])
    .size()
    .reset_index(name="content_count")
)

# Keep recent years
type_yearly = type_yearly[
    type_yearly["release_year"] >= 2008
]

print("\nMovies and TV Shows by Year:")
print(type_yearly.head(20))


# Separate Movies and TV Shows
movies = type_yearly[
    type_yearly["type"] == "Movie"
]

tv_shows = type_yearly[
    type_yearly["type"] == "TV Show"
]


# Movies vs TV Shows graph
plt.figure(figsize=(12, 6))

plt.plot(
    movies["release_year"],
    movies["content_count"],
    marker="o",
    label="Movies"
)

plt.plot(
    tv_shows["release_year"],
    tv_shows["content_count"],
    marker="o",
    label="TV Shows"
)

plt.title("Netflix Movies vs TV Shows Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Releases")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# STEP 5 — PREPARE DATA FOR MACHINE LEARNING
# ============================================================

X = yearly_releases[["release_year"]]

y = yearly_releases["content_count"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# ============================================================
# STEP 6 — TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(yearly_releases) * 0.8)

# Training data
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Testing data
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# STEP 7 — LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict test data
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# ============================================================
# STEP 8 — LINEAR REGRESSION EVALUATION
# ============================================================

# MAE
mae = mean_absolute_error(
    y_test,
    y_pred
)

print("\nMean Absolute Error (MAE):", mae)


# RMSE
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("Root Mean Squared Error (RMSE):", rmse)


# R2 Score
r2 = r2_score(
    y_test,
    y_pred
)

print("R2 Score:", r2)


# Actual vs Predicted table
comparison = pd.DataFrame({
    "Year": X_test["release_year"].values,
    "Actual": y_test.values,
    "Predicted": y_pred.round(0).astype(int)
})

print("\nActual vs Predicted:")
print(comparison)


# ============================================================
# STEP 9 — RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Random Forest
rf_model.fit(
    X_train,
    y_train
)

print("\nRandom Forest model trained successfully!")


# Random Forest predictions
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Predictions:")
print(rf_pred)


# ============================================================
# STEP 10 — RANDOM FOREST EVALUATION
# ============================================================

# Random Forest MAE
rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

print("\nRandom Forest MAE:", rf_mae)


# Random Forest RMSE
rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

print("Random Forest RMSE:", rf_rmse)


# Random Forest R2
rf_r2 = r2_score(
    y_test,
    rf_pred
)

print("Random Forest R2 Score:", rf_r2)


# ============================================================
# STEP 10.7 — MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        mae,
        rf_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse
    ],
    "R2 Score": [
        r2,
        rf_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(model_comparison)


# Save comparison
model_comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# ============================================================
# STEP 11 — MODEL COMPARISON GRAPHS
# ============================================================

# MAE comparison
plt.figure(figsize=(10, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["MAE"]
)

plt.title("MAE Comparison of Forecasting Models")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error")

plt.grid(axis="y")
plt.show()


# RMSE comparison
plt.figure(figsize=(10, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["RMSE"]
)

plt.title("RMSE Comparison of Forecasting Models")
plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error")

plt.grid(axis="y")
plt.show()


# R2 comparison
plt.figure(figsize=(10, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["R2 Score"]
)

plt.title("R2 Score Comparison of Forecasting Models")
plt.xlabel("Model")
plt.ylabel("R2 Score")

plt.grid(axis="y")
plt.show()


# ============================================================
# STEP 8 — SAVE LINEAR REGRESSION EVALUATION
# ============================================================

evaluation = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

evaluation.to_csv(
    "outputs/model_evaluation.csv",
    index=False
)

print("\nEvaluation results saved successfully!")


# ============================================================
# STEP 12 — FUTURE FORECAST
# ============================================================

# Future years
future_years = pd.DataFrame({
    "release_year": [
        2026,
        2027,
        2028,
        2029,
        2030
    ]
})

print("\nFuture Years:")
print(future_years)


# ------------------------------------------------------------
# Linear Regression future predictions
# ------------------------------------------------------------

future_predictions = model.predict(
    future_years
)

print("\nLinear Regression Future Predictions:")
print(future_predictions)


# ------------------------------------------------------------
# Random Forest future predictions
# ------------------------------------------------------------

rf_future_predictions = rf_model.predict(
    future_years
)

print("\nRandom Forest Future Predictions:")
print(rf_future_predictions)


# ------------------------------------------------------------
# Create comparison table
# ------------------------------------------------------------

future_results = future_years.copy()

future_results["Linear_Regression"] = (
    future_predictions
    .round(0)
    .astype(int)
)

future_results["Random_Forest"] = (
    rf_future_predictions
    .round(0)
    .astype(int)
)


print("\nFuture Forecast Comparison:")
print(future_results)


# ------------------------------------------------------------
# Prevent negative predictions
# ------------------------------------------------------------

future_results["Linear_Regression"] = (
    future_results["Linear_Regression"]
    .clip(lower=0)
)

future_results["Random_Forest"] = (
    future_results["Random_Forest"]
    .clip(lower=0)
)


print("\nFinal Future Forecast:")
print(future_results)


# ------------------------------------------------------------
# Save future forecast
# ------------------------------------------------------------

future_results.to_csv(
    "outputs/netflix_future_forecast_comparison.csv",
    index=False
)

print(
    "\nFuture forecast comparison saved successfully!"
)


# ============================================================
# FUTURE FORECAST GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    future_results["release_year"],
    future_results["Linear_Regression"],
    marker="o",
    linestyle="--",
    label="Linear Regression"
)

plt.plot(
    future_results["release_year"],
    future_results["Random_Forest"],
    marker="o",
    linestyle="--",
    label="Random Forest"
)

plt.title("Netflix Future Content Forecast Comparison")
plt.xlabel("Year")
plt.ylabel("Predicted Number of Releases")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# HISTORICAL + FUTURE FORECAST GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

# Historical data
plt.plot(
    yearly_releases["release_year"],
    yearly_releases["content_count"],
    marker="o",
    label="Historical"
)

# Linear Regression forecast
plt.plot(
    future_results["release_year"],
    future_results["Linear_Regression"],
    marker="o",
    linestyle="--",
    label="Linear Regression Forecast"
)

# Random Forest forecast
plt.plot(
    future_results["release_year"],
    future_results["Random_Forest"],
    marker="o",
    linestyle="--",
    label="Random Forest Forecast"
)

plt.title(
    "Netflix Historical and Future Content Trend"
)

plt.xlabel("Year")
plt.ylabel("Number of Releases")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    X_test["release_year"],
    y_test,
    marker="o",
    label="Actual"
)

plt.plot(
    X_test["release_year"],
    y_pred,
    marker="o",
    linestyle="--",
    label="Linear Regression Predicted"
)

plt.plot(
    X_test["release_year"],
    rf_pred,
    marker="o",
    linestyle="--",
    label="Random Forest Predicted"
)

plt.title(
    "Actual vs Predicted Netflix Releases"
)

plt.xlabel("Release Year")
plt.ylabel("Number of Releases")

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "models/netflix_trend_model.pkl"
)

print("\nLinear Regression model saved successfully!")


# Save Random Forest model
joblib.dump(
    rf_model,
    "models/netflix_random_forest_model.pkl"
)

print("Random Forest model saved successfully!")


print("\n========================================")
print("      NETFLIX FORECASTING COMPLETE")
print("========================================")

# ============================================================
# STEP 14 — FINAL MODEL ANALYSIS & INSIGHTS
# ============================================================

print("\n========================================")
print("       FINAL MODEL ANALYSIS")
print("========================================")


# Compare model performance
if rf_mae < mae:
    better_model = "Random Forest"
    better_mae = rf_mae
    better_rmse = rf_rmse
    better_r2 = rf_r2

else:
    better_model = "Linear Regression"
    better_mae = mae
    better_rmse = rmse
    better_r2 = r2


print("\nBest Model:", better_model)

print("Best Model MAE:", round(better_mae, 2))

print("Best Model RMSE:", round(better_rmse, 2))

print("Best Model R2 Score:", round(better_r2, 2))


# ============================================================
# FUTURE FORECAST INSIGHTS
# ============================================================

print("\n========================================")
print("       FUTURE FORECAST INSIGHTS")
print("========================================")


print("\nFuture Netflix Content Predictions:")

print(future_results)


# Linear Regression trend
linear_start = future_results["Linear_Regression"].iloc[0]
linear_end = future_results["Linear_Regression"].iloc[-1]

linear_change = linear_end - linear_start


print("\nLinear Regression Forecast:")

print(
    "Expected change from 2026 to 2030:",
    linear_change,
    "releases"
)


# Random Forest trend
rf_start = future_results["Random_Forest"].iloc[0]
rf_end = future_results["Random_Forest"].iloc[-1]

rf_change = rf_end - rf_start


print("\nRandom Forest Forecast:")

print(
    "Expected change from 2026 to 2030:",
    rf_change,
    "releases"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n========================================")
print("       FINAL CONCLUSION")
print("========================================")


if linear_change > 0:

    print(
        "Linear Regression predicts an increasing "
        "Netflix content trend from 2026 to 2030."
    )

elif linear_change < 0:

    print(
        "Linear Regression predicts a decreasing "
        "Netflix content trend from 2026 to 2030."
    )

else:

    print(
        "Linear Regression predicts a stable "
        "Netflix content trend."
    )


if rf_change > 0:

    print(
        "Random Forest predicts an increasing "
        "Netflix content trend."
    )

elif rf_change < 0:

    print(
        "Random Forest predicts a decreasing "
        "Netflix content trend."
    )

else:

    print(
        "Random Forest predicts a stable "
        "Netflix content trend."
    )


print("\nForecasting project analysis completed successfully!")