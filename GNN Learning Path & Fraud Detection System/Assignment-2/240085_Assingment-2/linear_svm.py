import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Regularization + hinge loss gradient
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * (-y_[idx])

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model parameters not initialized")
        linear_output = np.dot(X, self.w) - self.b
        return np.where(linear_output >= 0, 1, -1)