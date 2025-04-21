# Air Quality Index (AQI) Prediction using Machine Learning and Deep Learning

This project presents an end-to-end solution for predicting the **Air Quality Index (AQI)** using a combination of Machine Learning and Deep Learning techniques. By utilizing historical air quality data, the system aims to generate accurate forecasts that can help inform the public and assist policymakers in making proactive decisions to manage air pollution.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [System Design](#system-design)
- [Project Directory Structure](#project-directory-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Installation Guide](#installation-guide)
- [System Requirements](#system-requirements)
- [Conclusion](#conclusion)

---

## Introduction

### Purpose

The primary objective of this project is to develop a predictive system capable of forecasting AQI values using historical environmental data. The intended outcomes include:

- Leveraging ML/DL models to improve AQI prediction accuracy.
- Providing a reliable tool for both citizens and organizations to better prepare for air quality fluctuations.
- Offering a foundation for smarter environmental monitoring applications.

---

## Project Overview

### Problem Statement

Air pollution is one of the most significant health hazards in rapidly urbanizing regions. Major cities often experience hazardous air quality levels, which can have serious implications on public health, urban planning, and climate change. This project addresses this concern by creating a robust forecasting system that anticipates AQI changes using measurable environmental features.

### Objectives

- Design and implement multiple regression-based models for AQI prediction.
- Compare and evaluate performance metrics across algorithms.
- Build an interactive and user-friendly web interface using **Streamlit**.

---

## System Design

### High-Level Workflow

The system follows a structured pipeline:

1. **Data Collection**: Acquisition of historical air quality data from trusted sources.
2. **Data Cleaning**: Removal of missing values, anomalies, and irrelevant features.
3. **Feature Engineering**: Identification and transformation of key features influencing AQI.
4. **Model Training**: Development of multiple regression models including:
   - Linear Regression
   - Random Forest Regressor
   - XGBoost Regressor
   - Neural Networks
5. **Model Evaluation**: Assessment using statistical metrics such as MSE, RMSE, and R².
6. **Deployment**: Integration of the model into a web-based application for real-time predictions.

---

## Project Directory Structure

```bash
AQI/
├── src/
│   ├── aqi_predictor_nn_final.ipynb
│   ├── main.py
│   └── images/
├── data/
│   ├── city_aqi_day.csv
│   ├── city_hour.csv.zip
│   ├── clean_data.csv
│   └── no_missing.csv
├── exp/
│   ├── aqi_predictor_exp.ipynb
│   ├── images/
│   └── main.py
├── frontend/
│   └── main.py
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Data Preprocessing

### Data Cleaning

- Addressed missing values through statistical imputation or removal.
- Detected and eliminated outliers using **Z-score** and **Interquartile Range (IQR)** techniques.
- Ensured consistency in time-series formatting and unit conversions.

### Feature Engineering

- Removed features with high multicollinearity using VIF analysis.
- Derived new variables based on domain insights (e.g., pollutant ratios).
- Applied **Principal Component Analysis (PCA)** selectively to reduce feature dimensionality.

### Scaling and Splitting

- Scaled features using **MinMaxScaler** or **StandardScaler** depending on the model.
- Performed an 80/20 train-test split for generalizability testing.

---

## Model Training

### Algorithms Implemented

- **Linear Regression**: Established as a baseline model.
- **Random Forest Regressor**: Utilizes ensemble learning for higher accuracy.
- **XGBoost**: Employs gradient boosting techniques for improved prediction.
- **Neural Networks**: Comprises dense layers, ReLU activations, and dropout regularization.

### Training Strategy

- Hyperparameter tuning performed via **GridSearchCV** and **RandomizedSearchCV**.
- Implemented **k-Fold Cross Validation** to reduce the risk of overfitting.
- Utilized early stopping for deep learning models to optimize performance.

---

## Model Evaluation

### Evaluation Metrics

- **Mean Squared Error (MSE)**: Captures the average squared deviation.
- **Root Mean Squared Error (RMSE)**: Provides a direct interpretation in original units.
- **R² Score (Coefficient of Determination)**: Explains variance in predictions.

### Comparative Analysis

| Model              | MSE    | RMSE   | R² Score |
|--------------------|--------|--------|----------|
| Linear Regression  | 320.5  | 17.89  | 0.81     |
| Random Forest      | 210.3  | 14.50  | 0.89     |
| XGBoost            | 198.7  | 14.09  | 0.90     |
| Neural Network     | 185.2  | 13.61  | 0.91     |

---

## Deployment

### Frontend Interface

- Developed using **Streamlit** to facilitate real-time predictions.
- Allows user inputs for pollutant concentrations and displays the corresponding AQI.

### Running the Application

```bash
streamlit run DL_model/main.py
streamlit run exp/main.py
streamlit run frontend/main.py
```

### Backend Integration

- Trained models are serialized using `joblib` or `pickle`.
- Models are loaded dynamically in the Streamlit app to serve predictions efficiently.

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/sidj01sar26/AQI.git
cd AQI/
```

### Step 2: Create a Virtual Environment

```bash
# Using virtualenv
python -m venv venv
source venv/bin/activate

# OR using conda
conda create -n myenv python=3.10
conda activate myenv
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## System Requirements

### Software

- Python 3.8 or higher
- Jupyter Notebook / Lab (for development)
- Streamlit (for frontend)
- Required libraries:
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`, `seaborn`

### Hardware

- Processor: Intel Core i5 (or equivalent) or higher
- Memory: Minimum 8 GB RAM
- Storage: At least 2 GB of available space

---

## Conclusion

This project showcases a complete pipeline for AQI prediction, from data preprocessing to deployment. Through the application of Machine Learning and Deep Learning techniques, it achieves high predictive performance and user interactivity. The key advantages include:

- High accuracy and reliability in AQI forecasting.
- An intuitive web-based interface for end-users.
- Modular design, allowing for future extensions or integrations with IoT or real-time monitoring systems.

This system contributes toward smarter environmental monitoring and can be scaled further to support region-specific forecasting models.
