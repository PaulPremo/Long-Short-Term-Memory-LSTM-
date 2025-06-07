import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- IMPORTANT: Replace <ticker> with your desired stock symbol (e.g., 'AAPL', 'GOOG', 'MSFT') ---
STOCK_TICKER = 'AAPL' # Default to AAPL, change this as needed!
# --------------------------------------------------------------------------------------------------

# 1. Scarica i dati storici dello stock specificato dal ticker
df = yf.download(STOCK_TICKER, start='2015-01-01', end='2025-01-01')
data = df[['Close']]

# 2. Normalizza i dati
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Funzione per creare sequenze
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 4. Scegli la finestra temporale
window_size = 60 # Prova anche 30, 90, ecc.
X, y = create_sequences(scaled_data, window_size)

# 5. Dividi in training e test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Costruisci la rete LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Allena il modello
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 8. Previsioni
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
true_prices = scaler.inverse_transform(y_test)

# 9. Visualizza i risultati
plt.figure(figsize=(12, 6))
plt.plot(true_prices, label='Prezzi Reali')
plt.plot(predicted_prices, label='Prezzi Previsti')
plt.title(f'Previsione Prezzi {STOCK_TICKER} con LSTM') # Usa il ticker qui
plt.xlabel('Tempo')
plt.ylabel('Prezzo')
plt.legend()
plt.grid(True)
plt.show()
