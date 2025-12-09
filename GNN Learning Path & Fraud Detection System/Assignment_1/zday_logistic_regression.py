import numpy as np
import matplotlib.pyplot as plt

# parse the CSV file
def parse_csv(filename):
    X = []  
    y = []  
    with open(filename, 'r') as f:
        next(f)  # skip header
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                X.append([float(parts[0]), float(parts[1])])
                y.append(int(parts[2]))
    return np.array(X), np.array(y)

# Normalize features (z-score)
def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression with Gradient Descent
def logistic_regression(X, y, lr=0.1, iterations=1000):
    m, n = X.shape
    X_bias = np.hstack([np.ones((m, 1)), X])  # Add bias term
    theta = np.zeros(n + 1)
    cost_history = []
    for i in range(iterations):
        z = X_bias @ theta
        h = sigmoid(z)
        cost = -(1/m) * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
        grad = (1/m) * (X_bias.T @ (h - y))
        theta -= lr * grad
        cost_history.append(cost)
        if (i+1) % 100 == 0:
            print(f"Iter {i+1}: Cost={cost:.5f}")
    return theta, cost_history

def predict_proba(x, theta, mean, std):
    x_norm = (x - mean) / std
    x_input = np.hstack([1, x_norm])
    return sigmoid(x_input @ theta)

# Plotting functions
def plot_cost(cost_history):
    plt.figure(figsize=(7,4))
    plt.plot(cost_history, color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Loss)')
    plt.title('Cost Function Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('zday_cost_curve.png', dpi=300)
    plt.show()

def plot_decision_boundary(X, y, theta, mean, std):
    plt.figure(figsize=(7,6))
    # Plot data points
    survived = y == 1
    infected = y == 0
    plt.scatter(X[survived,0], X[survived,1], c='g', label='Survived', s=60)
    plt.scatter(X[infected,0], X[infected,1], c='r', label='Infected', s=60)
    # Decision boundary
    x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    # For each x1, solve for x2: theta0 + theta1*x1_norm + theta2*x2_norm = 0
    x1_norm = (x1_vals - mean[0]) / std[0]
    # x2_norm = -(theta0 + theta1*x1_norm)/theta2
    theta0, theta1, theta2 = theta
    x2_norm = -(theta0 + theta1 * x1_norm) / theta2
    x2_vals = x2_norm * std[1] + mean[1]
    plt.plot(x1_vals, x2_vals, 'b-', label='Decision Boundary')
    plt.xlabel('Sprint Speed (km/h)')
    plt.ylabel('Ammo Clips')
    plt.title('Z-Day Survival Logistic Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('zday_decision_boundary.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("="*60)
    # Load and normalize data
    X, y = parse_csv('zombies_data - Sheet1.csv')
    X_norm, mean, std = normalize_features(X)
    # Train logistic regression
    theta, cost_history = logistic_regression(X_norm, y, lr=0.1, iterations=1000)
    # Test prediction
    test_runner = np.array([25, 1])
    prob = predict_proba(test_runner, theta, mean, std)
    print("\nTest Prediction:")
    print(f"Runner: 25 km/h, 1 Ammo Clip => Survival Probability: {prob*100:.2f}%")
    # Plot cost curve
    plot_cost(cost_history)
    # Plot decision boundary
    plot_decision_boundary(X, y, theta, mean, std)
