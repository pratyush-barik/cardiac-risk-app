# Machine Learning for Early Cardiac Risk Detection

## Problem Statement

Heart disease is one of the leading causes of death globally. Early detection is critical for timely intervention and improved patient outcomes. However, traditional diagnosis can be time-consuming and resource-intensive.

This project aims to develop a machine learning-based system that predicts the risk of heart disease using patient clinical data, enabling fast, accurate, and cost-effective preliminary assessment.

---

## Project Overview

This project implements a web-based machine learning application that analyzes user-provided health indicators and predicts cardiac risk.

The system integrates multiple supervised learning models and combines their outputs using an ensemble approach to improve prediction reliability.

---

## Live Deployment

Access the live application here:

https://cardiac-risk-app.onrender.com/predict

---

## Features

* Predicts heart disease risk from clinical inputs
* Ensemble-based prediction using multiple models
* Displays prediction along with confidence score
* Lightweight and deployable web application
* Structured separation of training and deployment pipeline

---

## Machine Learning Approach

### Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)

### Ensemble Method

* Soft voting (equal-weight averaging of model predictions)

---

## Project Structure

```id="t1sznq"
cardiac-risk-app/
│
├── app/                    # Deployment layer (used by FastAPI & Render)
│   ├── main.py             # Backend application
│   ├── requirements.txt
│   ├── model/              # Trained models (.pkl files)
│   └── static/             # CSS files
│
├── training/               # Training pipeline (reference & reproducibility)
│   ├── train_model.py      # Script to train ML models
│   └── heart disease dataset.csv   # Dataset used for training
│
└── README.md
```

---

## Training Module

The `training/` folder contains the complete machine learning pipeline used to build the models.

### Contents

* **Dataset (`heart disease dataset.csv`)**
  Contains clinical features such as symptoms, lifestyle factors, and patient attributes used for training.

* **Training Script (`train_model.py`)**

  * Loads and preprocesses the dataset
  * Splits data into training and testing sets
  * Trains multiple ML models
  * Evaluates model performance
  * Saves trained models (`.pkl`) for deployment

### Purpose

This separation ensures:

* Reproducibility of results
* Clear distinction between training and deployment
* Ease of experimentation without affecting the deployed application

---

## Tech Stack

* **Backend:** FastAPI
* **Machine Learning:** Scikit-learn
* **Frontend:** HTML, CSS
* **Deployment:** Render
* **Data Processing:** Pandas, NumPy

---

## Installation & Running Locally

### 1. Clone the repository

```id="l6y1mk"
git clone https://github.com/pratyush-barik/cardiac-risk-app.git
cd cardiac-risk-app/app
```

### 2. Install dependencies

```id="br6p3y"
pip install -r requirements.txt
```

### 3. Run the application

```id="bx2m9m"
python -m uvicorn main:app --reload
```

### 4. Open in browser

```id="z8p3vh"
http://127.0.0.1:8000
```

---

## How It Works

1. User inputs clinical parameters through the web interface
2. Data is preprocessed and scaled
3. Each model generates a prediction
4. Predictions are combined using ensemble voting
5. Final risk classification and confidence score are displayed

---

## Model Performance

All models are evaluated using a stratified train-test split to maintain class balance. The ensemble method improves robustness and reduces dependence on any single model.

---

## Future Improvements (Core Engine Focus)

* Implement probability calibration for more reliable confidence scores
* Replace equal-weight voting with weighted or adaptive ensemble methods
* Introduce stacking (meta-learning) for improved predictive performance
* Perform feature importance analysis and selection
* Improve dataset quality with real-world medical data
* Address potential overfitting and enhance generalization

---

## Use Case

* Early-stage cardiac risk screening tool
* Demonstration of ML applications in healthcare
* Foundation for decision-support systems

---
