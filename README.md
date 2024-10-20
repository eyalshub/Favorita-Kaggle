# Corporación Favorita Store Sales Forecasting
## Overview
This project tackles the Corporación Favorita Store Sales challenge from Kaggle, aiming to predict sales for different stores and product families using historical sales data, transactions, oil prices, and holidays information. Time-series forecasting techniques are employed to handle the complex relationships between time, store-specific factors, and sales patterns.

## Table of Contents
1. Project Description
2. Installation and Setup
3. Data Description
4. Modeling Approach
5. File Descriptions
6. Results
7. Contributors
8. License
## Project Description
The project applies various machine learning and statistical techniques for sales prediction, focusing on:

- Data preprocessing and merging different datasets (oil prices, holidays, transactions).
- Handling missing data using time-series models (Prophet) and rolling means.
- Feature engineering (encoding categorical features, generating time-based features).
- Model evaluation using metrics like RMSE and cross-validation.
- Dimensionality reduction (PCA) for feature selection and improving model performance.
## Goals:
- Predict store sales for the next 16 days based on the historical dataset.
- Implement feature selection methods and normalization techniques.
- Compare various regression models to achieve the best prediction performance.
## Installation and Setup
Follow these steps to set up the project in your environment.

## Requirements:
The project requires Python 3.x. The necessary libraries are listed in the requirements.txt file. Alternatively, you can install the dependencies as follows:

bash
Copy code
!pip install kaggle
!pip install fitter
!pip install pyodbc
!pip install python-dotenv
!pip install prophet
!pip install catboost
!pip install category_encoders
!pip install shap
## Steps to Set Up:
1. Clone the repository or download the project files.
2. Kaggle API: To access the dataset from Kaggle, upload the kaggle.json file (API key) and place it in the correct directory as follows:
bash
Copy code
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
3. Download the dataset from Kaggle:
bash
Copy code
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d /content/store-sales-data
## Data Description
The project uses multiple datasets from the Corporación Favorita Store Sales competition:

- holidays_events.csv: Contains information on holidays and events affecting sales.
- oil.csv: Daily oil prices.
- transactions.csv: Number of daily transactions for each store.
- stores.csv: Metadata on each store, including city, state, and store type.
- train.csv: Historical sales data for different stores and products.
- test.csv: Test data for predicting sales (submission requires filling sales values).
## Data Preprocessing:
- Missing values in oil prices were handled using Prophet and rolling averages.
- Transactions and sales were aggregated on a monthly and daily basis.
- Time-based features like day of the week, month, and week of the year were created for each row in the dataset.
- Target Encoding was used for categorical variables like store number, city, state, and more.
## Modeling Approach
The project uses the following techniques:

### 1. Feature Engineering & Selection:
- Target Encoding for categorical features.
- Recursive Feature Elimination (RFE) to identify the most relevant features.
- Pearson and ANOVA tests to measure feature significance.
### 2. Models Applied:
- Linear Regression: Used as a baseline model.
- Random Forest Regressor: For boosting performance.
- LightGBM and CatBoost: Gradient boosting methods tailored for handling categorical features and large datasets.
- Prophet: For time-series forecasting.
### 3. Dimensionality Reduction:
- Principal Component Analysis (PCA) to reduce the feature space and improve computational efficiency.
### 4. Evaluation Metrics:
- **RMSE (Root Mean Squared Error)** : Used to evaluate the accuracy of the models.
- Cross-validation techniques were applied to prevent overfitting and ensure model robustness.
## File Descriptions
- Favorita.ipynb: Main Jupyter notebook containing the entire data pipeline, from loading and preprocessing to modeling and prediction.
- requirements.txt: List of Python dependencies required to run the project.
- train.csv, test.csv, holidays_events.csv, oil.csv, stores.csv, transactions.csv: Data files provided by Kaggle.
## Results
The RandomForestRegressor combined with feature scaling and PCA showed improved performance, with the best RMSE after applying:

- PCA for dimensionality reduction.
- Recursive Feature Elimination (RFE) for selecting the top features.
- Target Encoding for handling categorical variables.
Additional tuning and experimentation with gradient boosting models like LightGBM and CatBoost resulted in competitive results.

## Contributors
### Nadav Toledo – Data Engineer student
### Eyal Shubeli – Data Engineer student
- Kaggle Corporation for providing the dataset and the competition platform.
## License
This project is licensed under the MIT License. See the LICENSE file for details.
