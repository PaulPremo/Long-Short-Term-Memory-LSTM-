# Long-Short-Term-Memory-LSTM-
Stock Price Prediction with LSTM

---

## Project Description

This project implements a Recurrent Neural Network (RNN) using **Long Short-Term Memory (LSTM)** layers to predict the future closing prices of a given stock **ticker**. It leverages historical stock data fetched directly from Yahoo Finance, preprocesses it, trains an LSTM model, and then visualizes the predicted prices against the actual prices.

This project is ideal for understanding:
* How to download financial data programmatically for **any chosen stock ticker**.
* The process of **data normalization** for neural networks.
* Creating sequential data for time series forecasting.
* Building and training a basic **LSTM model** with Keras/TensorFlow.
* Visualizing model predictions against actual values.

## Features

* **Data Acquisition:** Automatically downloads historical stock data for a specified ticker using `yfinance`.
* **Data Preprocessing:** Scales data using `MinMaxScaler` for optimal neural network performance.
* **Sequence Generation:** Transforms time series data into sequences suitable for LSTM input.
* **LSTM Model:** A Sequential Keras model with a single LSTM layer for prediction.
* **Model Training:** Trains the LSTM model on a defined training set.
* **Prediction & Visualization:** Generates predictions on unseen data and plots them against actual prices for easy comparison.

## Requirements

Before running the code, ensure you have the following Python libraries installed:

* `yfinance`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `tensorflow` (or `keras`)

You can install them using pip:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
