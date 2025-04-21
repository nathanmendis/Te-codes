import numpy as np
import matplotlib.pyplot as plt
print("Natha Mendis TE-AIML 33543")
print("script started")
x = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")
plt.title("Activation Functions")
plt.legend()
plt.grid(True)
plt.show()
