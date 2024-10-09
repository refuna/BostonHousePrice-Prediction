import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the Boston housing dataset
from sklearn.datasets import load_boston

# Load and preprocess the dataset
def load_data():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    X = df.drop('PRICE', axis=1).values
    y = df['PRICE'].values
    return X, y

# Normalize the data
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

# Add bias term to the data
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.concatenate((ones, X), axis=1)
    return X_with_bias

# Calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Train the model using gradient descent
def train(X, y, learning_rate=0.01, num_iterations=1000):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0
    losses = []

    for _ in range(num_iterations):
        y_pred = np.dot(X, weights) + bias
        dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))  # Compute gradient for weights
        db = (1 / num_samples) * np.sum(y_pred - y)  # Compute gradient for bias
        weights -= learning_rate * dw  # Update weights with gradient descent
        bias -= learning_rate * db  # Update bias with gradient descent
        loss = mean_squared_error(y, y_pred)
        losses.append(loss)

    return weights, bias, losses

# Make predictions
def predict(X, weights, bias):
    y_pred = np.dot(X, weights) + bias
    return y_pred

# Visualize the training process
def plot_training_process(losses):
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Process')
    plt.show()

# Visualize the prediction results
def plot_predictions(y_true, y_pred, weights, bias):
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    sns.scatterplot(data=df, x='True', y='Predicted')

    # Draw the fitted line
    x_line = np.linspace(min(y_true), max(y_true), 100)
    y_line = weights[1] * x_line + bias  # Adjust the index in weights to match the corresponding feature
    plt.plot(x_line, y_line, color='r', label='Fitted Line')

    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('True vs Predicted Prices with Fitted Line')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load the dataset
    X, y = load_data()

    # Preprocess the data
    X_normalized = normalize(X)
    X_with_bias = add_bias(X_normalized)

    # Train the model
    weights, bias, losses = train(X_with_bias, y, learning_rate=0.01, num_iterations=1000)

    # Visualize the training process
    plot_training_process(losses)

    # Make predictions
    y_pred = predict(X_with_bias, weights, bias)

    # Visualize the prediction results (including fitted line)
    plot_predictions(y, y_pred, weights[1:], bias)

if __name__ == '__main__':
    main()
