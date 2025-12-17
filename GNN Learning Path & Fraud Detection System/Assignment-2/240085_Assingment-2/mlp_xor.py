import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class MLP_XOR:
    def __init__(self, n_hidden=2, lr=0.1, n_iters=10000):
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_iters = n_iters
        # Weights initialization
        self.W1 = np.random.randn(2, n_hidden)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1)
        self.b2 = np.zeros((1, 1))

    def fit(self, X, y):
        for i in range(self.n_iters):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = sigmoid(z2)

            # Backward pass
            dz2 = a2 - y
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * relu_deriv(z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update weights
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        return (a2 > 0.5).astype(int)

# Example
if __name__ == "__main__":
    mlp = MLP_XOR(n_hidden=2, lr=0.1, n_iters=10000)
    mlp.fit(X, y)
    preds = mlp.predict(X)
    print("Predictions:", preds.ravel())
    print("Ground truth:", y.ravel())
