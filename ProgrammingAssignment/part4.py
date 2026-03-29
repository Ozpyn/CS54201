# Write a linear regression analysis program that finds out the fitting line plot for the given data, and predicts the value of a dependent variable given the value of an independent variable. 

import random
import matplotlib.pyplot as plt

def generate_linear_data(n, slope, intercept, noise_level):

    for i in range(n):
        x = i
        noise = random.uniform(-noise_level, noise_level)
        y = slope * x + intercept + noise
        yield (x, y)

def regression_analysis(points):

    n = len(points)

    x_total = sum(x for x, y in points)
    y_total = sum(y for x, y in points)

    x_avg = x_total / n
    y_avg = y_total / n

    sigma_x = sum((x - x_avg) ** 2 for x, y in points) / n
    covariance_xy = sum((x - x_avg) * (y - y_avg) for x, y in points) / n

    slope = covariance_xy / sigma_x if sigma_x != 0 else 0
    intercept = y_avg - slope * x_avg

    return slope, intercept

def visualize_regression(points, slope, intercept):
    x_vals = [x for x, y in points]
    y_vals = [y for x, y in points]

    sorted_points = sorted(points)
    x_sorted = [x for x, y in sorted_points]
    y_pred = [slope * x + intercept for x in x_sorted]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_vals, color='blue', label='Data Points')
    plt.plot(x_sorted, y_pred, color='red', label='Regression Line')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression Visualization")
    plt.legend()
    plt.grid(True)

    plt.text(min(x_vals), max(y_vals),
             f"y = {slope:.2f}x + {intercept:.2f}",
             fontsize=10,
             color='green')

    plt.show()


points = list(generate_linear_data(
    n=50,
    slope=2,
    intercept=5,
    noise_level=2
))

points = list(generate_linear_data(10, slope=2, intercept=3, noise_level=0.5))

slope, intercept = regression_analysis(points)

print("Slope:", slope)
print("Intercept:", intercept)

visualize_regression(points, slope, intercept)