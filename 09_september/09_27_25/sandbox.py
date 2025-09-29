import numpy as np

# -------------------------------
# Generate synthetic dataset
# -------------------------------
np.random.seed(42)  # for reproducibility

# Inputs: 1D values in [-1, 1]
X = np.linspace(-1, 1, 200).reshape(-1, 1)

# True function with noise: y = x^5 + small Gaussian noise
y_true = X**5 + 0.05 * np.random.randn(*X.shape)


# -------------------------------
# Define activation functions
# -------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # Derivative of ReLU: 1 if x>0 else 0
    return (x > 0).astype(float)

def identity(x):
    return x

def identity_derivative(x):
    # Derivative of identity function = 1
    return np.ones_like(x)


# -------------------------------
# Layer class
# -------------------------------
class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        # Xavier initialization for weights
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

        # Activation function and its derivative
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, X):
        # Save input for use in backpropagation
        self.X = X
        # Linear combination
        self.Z = np.dot(X, self.W) + self.b
        # Apply activation function
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, learning_rate):
        """
        dA: gradient from the next layer (or loss function if output layer)
        """
        # Step 1: derivative of activation
        dZ = dA * self.activation_derivative(self.Z)

        # Step 2: compute gradients for weights and biases
        dW = np.dot(self.X.T, dZ) / self.X.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]

        # Step 3: compute gradient to pass to previous layer
        dX = np.dot(dZ, self.W.T)

        # Step 4: update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX


# -------------------------------
# Neural Network class
# -------------------------------
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes: list of layer dimensions, e.g. [1, 10, 10, 1]
        """
        self.layers = []
        self.learning_rate = learning_rate

        # Create hidden layers (ReLU) and output layer (identity)
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], relu, relu_derivative))

        # Last layer: identity activation for regression
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], identity, identity_derivative))

    def forward(self, X):
        # Pass data through all layers
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_pred, y_true):
        # Compute gradient of loss w.r.t predictions (MSE loss)
        # dLoss/dy_pred = 2 * (y_pred - y_true)
        dA = 2 * (y_pred - y_true) / y_true.shape[0]

        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.learning_rate)

    def train(self, X, y, epochs=1000, print_every=100):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute mean squared error
            loss = np.mean((y_true - y_pred)**2)

            # Backward pass
            self.backward(y_pred, y)

            # Print progress
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


# -------------------------------
# Train the model
# -------------------------------
nn = NeuralNetwork([1, 16, 16, 1], learning_rate=0.05)
nn.train(X, y_true, epochs=2000, print_every=200)


# -------------------------------
# Test the network
# -------------------------------
import matplotlib.pyplot as plt

y_pred = nn.forward(X)

plt.scatter(X, y_true, label="Noisy Data", alpha=0.6)
plt.plot(X, X**5, color="green", label="True function (x^5)")
plt.plot(X, y_pred, color="red", label="NN prediction")
plt.legend()
plt.show()
