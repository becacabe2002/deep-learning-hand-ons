import numpy as np

# --- Loss functions ---

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error -- measures the average squared difference between the predicted values and the actual values.
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of the Mean Squared Error loss function.
    """
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Cost function for binary classification problems, penalizes the model for incorrect predictions.
    binary_cross_entropy_loss = -1/N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) by clipping predictions
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_loss_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid division by zero
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size

# --- Activation functions ---
# Activation function is a mathematical function applied to the output of a layer in a neural network, introducing non-linearity to the model.

class Sigmoid:
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out =  1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        #  s * (1 -s)
        return grad_output * (self.out * (1 - self.out))
    
class Tanh:
    """
    Hyperbolic tangent activation function: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = tanh(x)
    Why use tanh?
    * Introduce non-linearity to the model, which allows it to learn complex patterns and relationships in the data.
    * The output of the tanh function is centered around zero, which can make the training more efficient and achieve faster convergence.
    * The tanh function can help mitigate the vanishing gradient problem (in some extent), especially when compared to the sigmoid function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # derivative of tanh is 1 - tanh^2(x)
        return grad_output * (1 - self.out ** 2)
# --- Linear Layer class ---

class LinearLayer:
    """
    A simple fully connected linear layer implementing y = xW + b
    """
    def __init__(self, in_features: int, out_features: int):
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.dw = None
        self.db = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        - dL/dW = x.T * dL/dy
        - dL/db = sum(dL/dy)
        - dL/dx = dL/dy * W.T
        """
        self.dw = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)

