# Write a linear regression analysis program that finds out the fitting line plot for the given data, and predicts the value of a dependent variable given the value of an independent variable. 

import random

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

data_gen = generate_linear_data(10, slope=2, intercept=3, noise_level=0.5)
points = list(data_gen)

slope, intercept = regression_analysis(points)

print("Slope:", slope)
print("Intercept:", intercept)