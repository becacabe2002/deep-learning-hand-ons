import numpy as np

class Optimizer:
    """
    Base class for optimizers
    """
    def __init__(self, params):
        self.params = params
    
    def step(self, grads):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum
    * Momentum helps the optimizer navigate through ravins and oscillations by adding a fraction of the previous update to the current update.
    """
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param) for param in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.momentum > 0:
                # update velocity = montomentum * velocity + (1 - momentum) * gradient
                self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * grad
                # update parameter = parameter - learning_rate * velocity
                param -= self.lr * self.velocities[i]
            else: 
                # basic SGD update without momentum
                param -= self.lr * grad

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer is an extension of SGD that computes adaptive learning rates for each parameter. It maintains two moving averages: one for the gradients (first moment) and one for the squared gradients (second moment). Adam combines the benefits of both momentum and RMSProp, making it well-suited for training deep neural networks.
    Reference [equations](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/)
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param) for param in params]  # first moment (mean)
        self.v = [np.zeros_like(param) for param in params]  # second moment (uncentered variance)
        self.t = 0  # time step

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second monent estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Since both m and v are biased towards zero in the initial steps, we need to correct it
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Final weight update
            param -= m_hat * self.lr / (np.sqrt(v_hat) + self.epsilon)
