# 🎬 Netflix Trend Forecasting

A Machine Learning project that analyzes historical Netflix content release patterns and forecasts future Netflix content releases using **Linear Regression** and **Random Forest Regression**.

The project also includes an interactive **Streamlit dashboard** for exploring historical trends, generating future forecasts, and comparing model performance.

---

## 📌 Project Overview

The **Netflix Trend Forecasting** project uses the Netflix Titles dataset to analyze content release patterns over the years.

The project:

* Analyzes yearly Netflix content releases
* Compares Movies and TV Shows
* Trains Machine Learning forecasting models
* Evaluates model performance
* Predicts future Netflix content releases
* Provides an interactive user interface
* Compares Linear Regression and Random Forest predictions

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical Netflix content release trends.
2. Identify yearly content release patterns.
3. Compare Movies and TV Shows over time.
4. Build Machine Learning models for trend forecasting.
5. Evaluate model performance using standard regression metrics.
6. Forecast future Netflix content releases.
7. Provide an interactive dashboard for users.

---

## 🧠 Machine Learning Models

Two regression models are used:

### 1. Linear Regression

Linear Regression is used to identify the relationship between the release year and the number of Netflix content releases.

### 2. Random Forest Regressor

Random Forest Regression is used as a second forecasting approach and compared with Linear Regression.

---

## 📊 Model Evaluation

The models are evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

The project generates a model comparison table containing these evaluation metrics.

---

## 🔮 Future Forecasting

The project generates Netflix content forecasts for future years.

The trained models can predict the expected number of content releases for a selected year.

The Streamlit interface allows users to select a forecast year and view predictions from both models.

---

## 🖥️ User Interface

The project includes a Streamlit-based interactive dashboard.

### Dashboard

The dashboard provides:

* Total Netflix titles
* Number of Movies
* Number of TV Shows
* Latest release year
* Historical Netflix release trend

### Historical Analysis

Users can explore:

* Yearly Netflix releases
* Movies vs TV Shows
* Historical release trends

### Future Forecast

Users can select a year and generate predictions using:

* Linear Regression
* Random Forest

### Model Comparison

Users can compare:

* MAE
* RMSE
* R² Score

for both Machine Learning models.

---

## 📁 Project Structure

```text
Netflix-Trend-Forecasting/
│
├── data/
│   └── netflix_titles.csv
│
├── models/
│   ├── netflix_trend_model.pkl
│   └── netflix_random_forest_model.pkl
│
├── outputs/
│   ├── yearly_netflix_releases.csv
│   ├── model_comparison.csv
│   ├── model_evaluation.csv
│   └── netflix_future_forecast_comparison.csv
│
├── ui/
│   └── app.py
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit

---

## 📂 Dataset

The project uses the **Netflix Titles Dataset**.

The dataset contains Netflix content information including:

* Title
* Type
* Release Year
* Other Netflix content attributes

The project primarily uses the `release_year` and `type` columns for trend analysis and forecasting.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/numeerayousaf-coder/Netflix-Trend-Forecasting.git
```

### 2. Open the project folder

```bash
cd Netflix-Trend-Forecasting
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Machine Learning Project

To train the models and generate the outputs:

```bash
python train.py
```

This will:

* Load the dataset
* Prepare the yearly release data
* Train Linear Regression
* Train Random Forest
* Evaluate the models
* Generate forecasts
* Save trained models
* Save output files

---

## 🔮 Run Prediction

The command-line prediction script can be executed using:

```bash
python predict.py
```

The program asks the user to enter a future year and displays the predicted number of Netflix content releases.

---

## 🖥️ Run the User Interface

To launch the Streamlit dashboard:

```bash
streamlit run ui/app.py
```

The dashboard will open in your web browser.

---

## 📈 Project Outputs

The project generates several output files:

### `yearly_netflix_releases.csv`

Contains yearly Netflix content release counts.

### `model_comparison.csv`

Contains the performance comparison of Linear Regression and Random Forest.

### `model_evaluation.csv`

Contains evaluation results for the Linear Regression model.

### `netflix_future_forecast_comparison.csv`

Contains future predictions from both forecasting models.

---

## 📊 Visualizations

The project includes visualizations for:

* Netflix Content Release Trend
* Movies vs TV Shows Release Trend
* MAE Model Comparison
* RMSE Model Comparison
* R² Score Comparison
* Future Forecast Comparison
* Historical vs Future Forecast
* Actual vs Predicted Values

---

## 💡 Key Features

* 📊 Historical Netflix trend analysis
* 🎬 Movies vs TV Shows comparison
* 🤖 Two Machine Learning models
* 📈 Model performance evaluation
* 🔮 Future content forecasting
* 🖥️ Interactive Streamlit dashboard
* 📁 Automatically generated output files
* 💾 Saved trained Machine Learning models

---

## 👩‍💻 Author

**Numeera Yousaf**

BS Software Engineering Student

GitHub:
https://github.com/numeerayousaf-coder

---

## 📌 Project Status

**Completed**

The project includes data analysis, Machine Learning model training, model evaluation, future forecasting, and an interactive Streamlit user interface.
