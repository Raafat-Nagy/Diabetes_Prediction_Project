# [Diabetes Classification Project](https://diabetes-prediction--project.streamlit.app)

## Project Overview

This project focuses on building and deploying a machine learning model to predict diabetes based on various health metrics. It includes a Jupyter notebook for data exploration, model training, and evaluation, as well as a Streamlit app for user-friendly predictions.

## Project Structure

```
Diabetes_Prediction_Project/
│
├── data/
│   └── Diabetes_Data.csv
│
├── notebooks/
│   └── Diabetes_Classification.ipynb
│
├── models/
│   ├── Diabetes_Model.pkl
│   └── Diabetes_StandardScaler.pkl
│
├── app/
│   ├── streamlit_app.py
│   └── utils.py
│
└── requirements.txt
```

## Features

- **Data Exploration and Visualization**: Comprehensive data analysis with visualizations to understand patterns and relationships.
- **Model Training**: Implementation of multiple classification models with hyperparameter tuning using `RandomizedSearchCV`.
- **Model Evaluation**: Detailed evaluation metrics including accuracy, F1-score, precision, and recall.
- **Streamlit Application**: An interactive web app for predicting diabetes based on user inputs.

## Data Description

The dataset used for training the models includes the following features:

- **Pregnancies**: Number of times pregnant (For Female).
- **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)²).
- **DiabetesPedigreeFunction**: Diabetes pedigree function.
- **Age**: Age (years).
- **Outcome**: Class variable (0 or 1), where 1 indicates diabetes and 0 indicates no diabetes.

## Installation

To get started with this project, clone the repository and install the required packages.

```bash
git clone <repository-url>
cd Diabetes_Prediction_Project
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

1. Navigate to the `app` directory:
    ```bash
    cd app
    ```
2. Start the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and contributions are welcome!
