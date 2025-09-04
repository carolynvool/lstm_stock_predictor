# Stock Data Predictor with LSTMs

This project implements a Long Short-Term Memory (LSTM) neural network to forecast next-day **open stock prices** using historical data from the S&P 500.

---

## Description
We use the Kaggle S&P 500 dataset and Yahoo Finance (`yfinance`) to build rolling windows of historical open prices (10, 20, 50 trading days) and train a custom PyTorch LSTM model. The model fuses gate computations into two matrix multiplications for efficiency.  

Results show that:  
- Window sizes up to ~20 days improve accuracy  
- MSE decreases steadily with training  
- R² on the test set reaches 0.96 for 20-day windows  

The project also highlights common pitfalls in financial forecasting (data leakage, temporal splits, scaling) and discusses possible extensions such as multi-feature inputs and attention mechanisms.

---

## Features
- Data preprocessing: normalization, rolling windows, train/val/test splits  
- Custom LSTM implementation in PyTorch  
- Training pipeline with Adam optimizer  
- Evaluation using MSE and R² metrics  
- Visualizations of predictions vs actuals and loss curves  

---

## Results (Summary)

| Window Size | Final Train Loss | Final Val Loss | R² (Test) |
|-------------|------------------|----------------|-----------|
| 10          | 0.0067           | 0.0029         | 0.94      |
| 20          | 0.0020           | 0.0014         | 0.96      |
| 50          | 0.0021           | 0.0015         | 0.95      |

---

## Authors
- Anjani Bahl – B.Sc. Management and Technology – 03767337 – anjani.bahl@tum.de  
- Carolyn Vool – B.Sc. Management and Technology – 03775411 – carolyn.vool@tum.de  
- Stanisław Woźniak – B.Sc. Management and Technology – 03777776 – ge92taq@mytum.de  

---

## References
- piEsposito, *PyTorch-LSTM-by-Hand: LSTM.ipynb*, GitHub Repository  
- Pilla, Prashant, and Raji Mekonen. *Forecasting S&P 500 Using LSTM Models*. arXiv, 2025  
