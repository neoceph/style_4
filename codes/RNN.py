import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Generate sine wave time series
# -----------------------------
t = np.linspace(0, 50, 2000)
data = np.sin(t)

# Create sequences
SEQ_LEN = 20

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = make_sequences(data, SEQ_LEN)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# -----------------------------
# Simple RNN model
# -----------------------------
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training
# -----------------------------
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    preds = model(X).numpy()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(data[SEQ_LEN:], label="True Sine")
plt.plot(preds, "--", label="RNN Prediction")
plt.legend()
plt.title("RNN Predicting a Sine Wave (Time-Series View)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(True)
plt.show()
