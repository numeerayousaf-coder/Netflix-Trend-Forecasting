import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/netflix_trend_model.pkl")

# Ask user for future year
year = int(input("Enter the year you want to predict: "))

# Create input data
future_year = pd.DataFrame({
    "release_year": [year]
})

# Generate prediction
prediction = model.predict(future_year)

# Prevent negative prediction
prediction = max(0, round(prediction[0]))

# Display result
print("\nNetflix Content Forecast")
print("------------------------")
print("Year:", year)
print("Predicted Content Releases:", prediction)