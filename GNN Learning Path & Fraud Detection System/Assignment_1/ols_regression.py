import numpy as np
import matplotlib.pyplot as plt

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

# Calculate OLS parameters (slope and intercept)
def calculate_ols_parameters(x, y):
    n = len(x)
    
    # Calculate sums
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x_squared = sum(x[i] ** 2 for i in range(n))
    
    # slope (m)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    
    # y-intercept (b)
    b = (sum_y - m * sum_x) / n
    
    return m, b

# Prediction function
def predict_price(square_feet, m, b):
    return m * square_feet + b

# R-squared for model evaluation
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
    print("=" * 60)
    
    square_footage, prices = parse_csv('housing_prices - housing_prices.csv')
    
    print(f"\nData loaded: {len(square_footage)} data points")
    print(f"Square Footage range: {min(square_footage)} - {max(square_footage)} sq ft")
    print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    
    # Calculate OLS parameters
    m, b = calculate_ols_parameters(square_footage, prices)
    
    print("\n" + "-" * 60)
    print("MODEL PARAMETERS (Line of Best Fit)")
    print("-" * 60)
    print(f"Slope (m): {m:.6f}")
    print(f"Y-intercept (b): {b:.6f}")
    
    # R-squared
    r_squared = calculate_r_squared(square_footage, prices, m, b)
    print(f"R-squared: {r_squared:.6f}")
    
    # prediction for 2,500 square feet
    target_sqft = 2500
    predicted_price = predict_price(target_sqft, m, b)
    
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"For a house with {target_sqft:,} square feet:")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print("=" * 60)
    
    # Visualization
    plt.figure(figsize=(10, 6))

    plt.scatter(square_footage, prices, color='blue', alpha=0.6, s=50, label='Actual Data')
    x_line = np.linspace(min(square_footage), max(square_footage), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red', linewidth=2, label='OLS Line of Best Fit')
    
    # Highlight prediction point
    plt.scatter([target_sqft], [predicted_price], color='green', s=200, 
                marker='*', zorder=5, label=f'Prediction: {target_sqft} sq ft')
    
    plt.xlabel('Square Footage', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('OLS Regression: Housing Price Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('ols_regression_plot.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'ols_regression_plot.png'")
    
    plt.show()
