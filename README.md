# Customer-Churn-Prediction

# Customer Churn Prediction with ANN

This project uses an Artificial Neural Network (ANN) to predict customer churn. The model is built and deployed using Streamlit for an interactive web application.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)

## Overview
Customer churn prediction is crucial for businesses to retain customers by identifying those at risk of leaving. This project implements an ANN to predict whether a customer will churn based on various features.

## Dataset
The dataset contains the following columns:
- `creditscore`: Credit score of the customer
- `geography`: Customer's geography
- `gender`: Customer's gender
- `age`: Customer's age
- `tenure`: Number of years the customer has been with the bank
- `balance`: Account balance
- `numofproducts`: Number of products the customer has with the bank
- `hascrcard`: Whether the customer has a credit card (1 = Yes, 0 = No)
- `isactivemember`: Whether the customer is an active member (1 = Yes, 0 = No)
- `estimatedsalary`: Estimated salary of the customer

## Features
The features used for prediction are:
- `creditscore`
- `geography` (One-hot encoded)
- `gender` (Label encoded)
- `age`
- `tenure`
- `balance`
- `numofproducts`
- `hascrcard`
- `isactivemember`
- `estimatedsalary`

## Model Architecture
  The ANN model has the following architecture:
  - Input layer with 64 neurons and ReLU activation
  - Hidden layer with 32 neurons and ReLU activation
  - Output layer with 1 neuron and Sigmoid activation

## Requirements
  The required Python libraries are listed in the `requirements.txt` file. To install the dependencies, run:
    ```sh
    
    pip install -r requirements.txt

## Results
  The model predicts the likelihood of customer churn with a probability score. Based on this score, it determines if the customer is likely to churn or not.

## Contributing
  Contributions are welcome! Please fork this repository and submit a pull request for any improvements.
