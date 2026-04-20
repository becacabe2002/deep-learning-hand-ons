import numpy as np
from model import *
from numpy_from_scratch.gradient_checking.grad_check import compute_numerical_gradient, gradient_check

def generate_linear_data(n_samples=100):
    np.random.seed(42)
    X = np.random.rand(n_samples, 1)
    y = 3 * X + 1 + np.random.randn(n_samples, 1) * 0.3
    # expected parameters: w = 3, b = 1
    return X, y

def generate_classification_data(n_samples=100):
    np.random.seed(42)
    X1 = np.random.randn(n_samples // 2, 2)  + np.array([2, 2])
    X2 = np.random.randn(n_samples // 2, 2)  + np.array([-2, -2])
    X = np.vstack((X1, X2))
    y = np.vstack([np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))])
    return X, y

def train_linear_regression(input_features, input_labels, learning_rate=0.01, n_epochs=50):
    print("Training Linear Regression...")
    X, y = input_features, input_labels
    model = LinearLayer(in_features=X.shape[1], out_features=1)
    for epoch in range(n_epochs):
        # forward pass
        y_pred = model.forward(X)
        loss = mse_loss(y_pred, y)

        # backward pass
        grad_y = mse_loss_grad(y_pred, y)
        model.backward(grad_y)

        # update parameters
        model.weights -= learning_rate * model.dw
        model.bias -= learning_rate * model.db

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")
        print(f"Final parameters: w = {model.weights.flatten()[0]:.4f}, b = {model.bias.flatten()[0]:.4f}\n")

def train_binary_classification(input_features, input_labels, learning_rate=0.01, n_epochs=50):
    print("Training Binary Classification...")
    X, y = input_features, input_labels
    model = LinearLayer(in_features=X.shape[1], out_features=1)
    sigmoid = Sigmoid()

    for epoch in range(n_epochs):
        # forward pass
        linear_out = model.forward(X)
        y_pred = sigmoid.forward(linear_out)
        loss = binary_cross_entropy_loss(y_pred, y)

        # backward pass
        grad_y = binary_cross_entropy_loss_grad(y_pred, y)
        grad_linear_out = sigmoid.backward(grad_y)
        model.backward(grad_linear_out)

        # update parameters
        model.weights -= learning_rate * model.dw
        model.bias -= learning_rate * model.db

        if (epoch + 1) % 10 == 0:
            predictions = (y_pred > 0.5).astype(int)
            # print("Y_pred (probabilities):", y_pred.flatten())
            # print("Predictions:", predictions.flatten())
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def check_regression_gradients(input_features, input_labels):
    print("Checking gradients...")
    X, y = input_features, input_labels
    model = LinearLayer(in_features=X.shape[1], out_features=1)

    # forward pass
    y_pred = model.forward(X)
    grad_y = mse_loss_grad(y_pred, y)
    model.backward(grad_y)

    dw_analytical = model.dw

    # Gradient checking
    def loss_fn(w_new):
        old_w = model.weights.copy()
        model.weights = w_new
        y_pred = model.forward(X)
        loss = mse_loss(y_pred, y)
        model.weights = old_w  
        return loss

    is_correct, relative_error = gradient_check(loss_fn, model.weights, dw_analytical)
    print(f"Gradient check passed: {is_correct}, Relative error: {relative_error:.2e}\n")

def check_classification_gradients(input_features, input_labels):
    print("Checking gradients for classification...")
    X, y = input_features, input_labels
    model = LinearLayer(in_features=X.shape[1], out_features=1)
    sigmoid = Sigmoid()

    # forward pass
    linear_out = model.forward(X)
    y_pred = sigmoid.forward(linear_out)
    grad_y = binary_cross_entropy_loss_grad(y_pred, y)
    grad_linear_out = sigmoid.backward(grad_y)
    model.backward(grad_linear_out)

    dw_analytical = model.dw

    # Gradient checking
    def loss_fn(w_new):
        old_w = model.weights.copy()
        model.weights = w_new
        linear_out = model.forward(X)
        y_pred = sigmoid.forward(linear_out)
        loss = binary_cross_entropy_loss(y_pred, y)
        model.weights = old_w  
        return loss

    is_correct, relative_error = gradient_check(loss_fn, model.weights, dw_analytical)
    print(f"Gradient check passed: {is_correct}, Relative error: {relative_error:.2e}\n")

if __name__ == "__main__":
    X_5, y_5 = generate_linear_data(5)
    check_regression_gradients(X_5, y_5)
    X, y = generate_linear_data(100)
    train_linear_regression(X, y, learning_rate=0.01, n_epochs=5000) # just for demonstration, too overfit

    X_5_cls, y_5_cls = generate_classification_data(5)
    check_classification_gradients(X_5_cls, y_5_cls)
    X_cls, y_cls = generate_classification_data(100)
    train_binary_classification(X_cls, y_cls, learning_rate=0.01, n_epochs=500)


