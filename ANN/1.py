import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# X-axis values
x = np.linspace(-10, 10, 400)

# Create subplots: 2 rows, 2 columns
plt.figure(figsize=(12, 8))

# 1. Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), color='blue')
plt.title("Sigmoid")
plt.grid(True)

# 2. Tanh
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), color='green')
plt.title("Tanh")
plt.grid(True)

# 3. ReLU
plt.subplot(2, 2, 3)
plt.plot(x, relu(x), color='red')
plt.title("ReLU")
plt.grid(True)

# 4. Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x), color='purple')
plt.title("Leaky ReLU")
plt.grid(True)

plt.tight_layout()
plt.show()
