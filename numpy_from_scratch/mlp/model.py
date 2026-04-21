import numpy as np
from numpy_from_scratch.linear_nn.model import LinearLayer

class ReLu:
    """
    Rectified Linear Unit (ReLU) activation function: f(x) = max(0, x)
    * Help maintain non-linearity in the model
    * Reduces the likelihood of vanishing gradients
    """
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)
    
class LeakyReLu:
    """
    Leaky ReLU activation function: f(x) = max(0.01x, x)
    * Similar to ReLU but allows a small, non-zero gradient when the input is negative, which can help mitigate the "dying ReLU" problem.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output):
        return grad_output * np.where(self.input > 0, 1, self.alpha)
    
class SoftMaxCrossEntropyLoss:
    """
    Softmax activation function combined with Cross-Entropy loss for multi-class classification problems.
    * The softmax function converts raw scores (logits) into probabilities, while the cross-entropy loss measures the difference between the predicted probabilities and the true labels.
    Softmax f(x_i) = exp(x_i)/ sum(exp(x_j)) for j in all classes
    Cross-Entropy Loss L = -sum(y_true * log(y_pred)) for all classes
    """

    def forward(self, logits, y_true):
        self.y_true = y_true
        # since softmax is shift-invariant, we can subtract the max logit from each logit to mitigate numerical instability (overflow).
        # Ex: e^1000 is inf.
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        batch_size = y_true.shape[0]
        if y_true.ndim == 1: # 1D lables
            log_likelihood = -np.log(self.probs[range(batch_size), y_true] + 1e-15) # only take the log probability of the correct class for each sample
        else: # one-hot encoded labels
            log_likelihood = -np.sum(y_true * np.log(self.probs + 1e-15), axis=1)
        
        return np.mean(log_likelihood)


    def backward(self):
        batch_size = self.y_true.shape[0]
        grad = self.probs.copy()

        if self.y_true.ndim == 1:
            grad[range(batch_size), self.y_true] -= 1
        else:
            grad = grad - self.y_true

        return grad / batch_size

class MLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output