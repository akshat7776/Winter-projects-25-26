import numpy as np
import matplotlib.pyplot as plt

# Step 1: Manually parse the CSV file
def parse_csv(filename):
    square_footage = []
    prices = []
    
    with open(filename, 'r') as file:
        next(file)
        
        # Parse each line
        for line in file:
            line = line.strip()
            if line: 
                parts = line.split(',')
                square_footage.append(float(parts[0]))
                prices.append(float(parts[1]))
    
    return square_footage, prices

# Normalize features for better gradient descent convergence
def normalize_features(x):
    mean = sum(x) / len(x)
    std = (sum((xi - mean) ** 2 for xi in x) / len(x)) ** 0.5
    x_normalized = [(xi - mean) / std for xi in x]
    return x_normalized, mean, std

def normalize_target(y):
    mean = sum(y) / len(y)
    std = (sum((yi - mean) ** 2 for yi in y) / len(y)) ** 0.5
    y_normalized = [(yi - mean) / std for yi in y]
    return y_normalized, mean, std

# Gradient Descent algorithm
def gradient_descent(x, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    n = len(x)
    
    # Initialize parameters
    m = 0.0  # slope
    b = 0.0  # y-intercept
    
    # Store history for visualization
    cost_history = []
    m_history = []
    b_history = []
    
    # Gradient descent iterations
    for iteration in range(iterations):
        # Calculate predictions
        y_pred = [m * x[i] + b for i in range(n)]
        
        # Calculate errors
        errors = [y_pred[i] - y[i] for i in range(n)]
        
        # Calculate cost (Mean Squared Error)
        cost = sum(error ** 2 for error in errors) / (2 * n)
        cost_history.append(cost)
        m_history.append(m)
        b_history.append(b)
        
        # Calculate gradients
        gradient_m = sum(errors[i] * x[i] for i in range(n)) / n
        gradient_b = sum(errors) / n
        
        # Update parameters
        m_new = m - learning_rate * gradient_m
        b_new = b - learning_rate * gradient_b
        
        # Check for convergence
        if abs(m_new - m) < tolerance and abs(b_new - b) < tolerance:
            print(f"Converged at iteration {iteration + 1}")
            m = m_new
            b = b_new
            break
        
        m = m_new
        b = b_new

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Cost = {cost:.6f}, m = {m:.6f}, b = {b:.6f}")
    
    return m, b, cost_history, m_history, b_history

# Denormalize parameters to get actual slope and intercept
def denormalize_parameters(m_norm, b_norm, x_mean, x_std, y_mean, y_std):
    m = m_norm * (y_std / x_std)
    b = y_mean - m * x_mean + b_norm * y_std
    return m, b

# Create prediction function
def predict_price(square_feet, m, b):
    return m * square_feet + b

# Calculate R-squared for model evaluation
def calculate_r_squared(x, y, m, b):
    y_mean = sum(y) / len(y)
    
    # Total sum of squares
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
    
    # Residual sum of squares
    ss_res = sum((y[i] - (m * x[i] + b)) ** 2 for i in range(len(y)))
    
    # R-squared
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

# Main execution
if __name__ == "__main__":
    # Parse the CSV file
    print("=" * 60)
    print("GRADIENT DESCENT REGRESSION - MANUAL IMPLEMENTATION")
    print("=" * 60)
    
    square_footage, prices = parse_csv('housing_prices - housing_prices.csv')
    
    print(f"\nData loaded: {len(square_footage)} data points")
    print(f"Square Footage range: {min(square_footage)} - {max(square_footage)} sq ft")
    print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    
    # Normalize features for better convergence
    print("\nNormalizing features for gradient descent...")
    x_normalized, x_mean, x_std = normalize_features(square_footage)
    y_normalized, y_mean, y_std = normalize_target(prices)
    
    # Run gradient descent

    print("-" * 60)
    learning_rate = 0.1
    iterations = 1000
    
    m_norm, b_norm, cost_history, m_history, b_history = gradient_descent(
        x_normalized, y_normalized, 
        learning_rate=learning_rate, 
        iterations=iterations
    )
    
    # Denormalize parameters to get actual slope and intercept
    m, b = denormalize_parameters(m_norm, b_norm, x_mean, x_std, y_mean, y_std)
    
    print("\n" + "-" * 60)
    print("GRADIENT DESCENT RESULTS")
    print("-" * 60)
    print(f"Slope (m): {m:.6f}")
    print(f"Y-intercept (b): {b:.6f}")
    
    # Calculate R-squared
    r_squared = calculate_r_squared(square_footage, prices, m, b)
    print(f"R-squared: {r_squared:.6f}")
    
    # Make prediction for 2,500 square feet
    target_sqft = 2500
    predicted_price = predict_price(target_sqft, m, b)
    
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"For a house with {target_sqft:,} square feet:")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print("=" * 60)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with regression line
    ax1 = axes[0]
    ax1.scatter(square_footage, prices, color='blue', alpha=0.6, s=50, label='Actual Data')
    
    x_line = np.linspace(min(square_footage), max(square_footage), 100)
    y_line = m * x_line + b
    ax1.plot(x_line, y_line, color='red', linewidth=2, label='Gradient Descent Line')
    
    ax1.scatter([target_sqft], [predicted_price], color='green', s=200, 
                marker='*', zorder=5, label=f'Prediction: {target_sqft} sq ft')
    
    ax1.set_xlabel('Square Footage', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Gradient Descent: Housing Price Prediction', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost function convergence
    ax2 = axes[1]
    ax2.plot(range(len(cost_history)), cost_history, color='purple', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost (MSE)', fontsize=12)
    ax2.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_plot.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'gradient_descent_plot.png'")
    
    plt.show()
    
    # Additional info
    print("\n" + "-" * 60)
    print("ALGORITHM DETAILS")
    print("-" * 60)
    print(f"Learning Rate: {learning_rate}")
    print(f"Total Iterations: {len(cost_history)}")
    print(f"Final Cost: {cost_history[-1]:.6f}")
    print("-" * 60)
