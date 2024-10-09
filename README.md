# BostonHousePrice-Prediction

## Project Overview

This project demonstrates the implementation of a linear regression model from scratch using the **Boston Housing Dataset**. The model predicts house prices based on several features such as crime rate, tax rate, proximity to the Charles River, and more. This project focuses on predicting house prices using a linear regression model trained through gradient descent. The entire process, from data loading and normalization to model training and evaluation, is implemented from scratch without using any machine learning libraries such as `scikit-learn` for the core logic.

## Dataset

The **Boston Housing Dataset** is used for this project. This dataset is included in `scikit-learn` and contains 506 instances, with 13 features describing various aspects of houses in Boston, such as crime rate, tax rate, number of rooms, etc.

### Dataset Details:
- **Features**: 13 continuous attributes (e.g., crime rate, number of rooms)
- **Target**: House price (in thousands of dollars)
- **Size**: 506 data points

## Features

The dataset contains the following features:
- `CRIM`: Per capita crime rate by town
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX`: Nitric oxides concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built prior to 1940
- `DIS`: Weighted distances to five Boston employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Full-value property tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town
- `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- `LSTAT`: Percentage of lower status of the population

## Model

The project builds a linear regression model from scratch. It includes:

- **Data Normalization**: Features are normalized to have zero mean and unit variance.
- **Gradient Descent**: The weights and bias are optimized using gradient descent to minimize the mean squared error.
- **Mean Squared Error**: Used as the loss function to evaluate model performance.

