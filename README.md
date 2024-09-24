# Comparative Analysis of Classification Algorithms to Predict Diabetic Individuals

A comparative analysis of three classification algorithms: k-Nearest Neighbors (k-NN), Support Vector Machine (SVM), and Neural Networks (Feed Forward). The analysis uses the `diabetes.csv` dataset to determine which algorithm yields the least classification error.

## Overview

The Jupyter notebook, `diabetes_classification.ipynb`, contains a detailed analysis of the following models:

1. **k-Nearest Neighbors (k-NN)**:
    - Evaluates model performance at various values of `k` to observe the effect on classification accuracy.
    
2. **Support Vector Machine (SVM)**:
    - Tests multiple kernel functions (linear, multi-polynomial, RBF) to find the best-performing kernel.
    
3. **Neural Networks**:
    - Implements a simple feedforward neural network using a multi-layer perceptron (MLP) to classify the diabetes dataset.

The goal of the analysis is to compare the models in terms of classification error and determine the most effective algorithm for this dataset.

## Dataset

- **File**: `diabetes.csv`
- **Source**: Pima Indians Diabetes Database (available on UCI Machine Learning Repository)
- **Features**:
    - `Pregnancies`: Number of times pregnant
    - `Glucose`: Plasma glucose concentration
    - `BloodPressure`: Diastolic blood pressure (mm Hg)
    - `SkinThickness`: Triceps skinfold thickness (mm)
    - `Insulin`: 2-Hour serum insulin (mu U/ml)
    - `BMI`: Body mass index (weight in kg/(height in m)^2)
    - `DiabetesPedigreeFunction`: A function that scores the likelihood of diabetes based on family history
    - `Age`: Age in years
    - `Outcome`: Binary variable indicating if the patient has diabetes (1) or not (0)

## Prerequisites

The following Python packages are required to run the notebook:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pytorch`
