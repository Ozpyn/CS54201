# Write a multi hidden layered (with number of hidden layers as parameter) neural network program with m input variables and n output variables. The size m, n and number of perceptrons in hidden layers are parameterized. Initially use random number function to give weights to the edges. Generate your test data (meaningfully) using a simple algorithm, and then test it out on neural network. There must be at least 100 test data for training.

import numpy as np

def mse(setA, setB):
    return np.mean((setA - setB) ** 2)
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]))
            self.biases.append(np.random.randn(1, self.layers[i+1]))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            self.z_values.append(z)
            a = sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y, learning_rate=0.1):
        output = self.activations[-1]
        error = y - output
        deltas = [error * sigmoid_derivative(output)]

        for i in reversed(range(len(self.layers) - 2)):
            delta = deltas[0].dot(self.weights[i+1].T) * sigmoid_derivative(self.activations[i+1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 100 == 0:
                loss = mse(y, self.activations[-1])
                # loss = np.mean((y - self.activations[-1])**2)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

m = 3   # input
n = 2   # output
data_size = 200
hidden_layers = [5, 4] # size() is number of layers, content is neurons in layer
learning_rate=0.175
epochs=4000
seed = 69

nn = NeuralNetwork(input_size=m, output_size=n, hidden_layers=hidden_layers)

np.random.seed(seed)

trunc = 1e4

# output = [mean(inputs), product(inputs)]
X_train = np.trunc(np.random.rand(data_size, m) * trunc) / trunc
y_train = np.zeros((data_size, n))
y_train[:, 0] = np.trunc((np.sum(X_train, axis=1) / m) * trunc) / trunc
y_train[:, 1] = np.trunc(np.prod(X_train, axis=1) * trunc) / trunc

nn.train(X_train, y_train, epochs, learning_rate)

X_test = np.array([[0.2, 0.5, 0.1],
                   [0.9, 0.3, 0.4]])
                   
X_test = np.trunc(np.random.rand(int(data_size / 10), m) * trunc) / trunc
y_pred = nn.predict(X_test)

print("\nTest Predictions:")
for i, x in enumerate(X_test):
    pred_str = ", ".join(f"{val}" for val in y_pred[i])
    print(f"Input: [{x}], Predicted Output: [{pred_str}]")

print("\nReal:")
y_calc = np.zeros((X_test.shape[0], n))
y_calc[:, 0] = np.sum(X_test, axis=1) / m
y_calc[:, 1] = np.prod(X_test, axis=1)

for i, x in enumerate(X_test):
    calc_str = ", ".join(f"{val}" for val in y_calc[i])
    print(f"Input: [{x}], Real Output: [{calc_str}]")

error = mse(y_calc, y_pred)
print(f"\nMean Squared Error on Test Set: {error:.8f}")