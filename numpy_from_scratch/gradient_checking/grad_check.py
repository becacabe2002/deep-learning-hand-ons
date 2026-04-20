import numpy as np

def compute_numerical_gradient(f, x, epsilon=1e-7):
    """
    Compute the numerical gradient of function f at point x using central difference formula.
    Central difference formula: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2 * ε)
    where ε is a small perturbation value.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + epsilon
        f_plus = f(x)
        x[idx] = old_val - epsilon
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = old_val  # restore original value
        it.iternext()
    return grad

def gradient_check(f, x, analytical_grad, epsilon=1e-7,threshold=1e-5) -> tuple[bool, float]:
    """
    Compute the relative error between the analytical gradient and the numerical gradient.
    Difference = ||analytical_grad - numerical_grad|| / (||analytical_grad|| + ||numerical_grad||)"""
    numerical_grad = compute_numerical_gradient(f, x, epsilon)
    
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)

    if denominator == 0:
        return 0.0
    
    relative_error = numerator / denominator
    return relative_error < threshold, relative_error

if __name__ == "__main__":
    # Example usage
    def simple_quadratic(x):
        return np.sum(x ** 2)

    x = np.array([1.0, 2.0, 3.0])
    analytical_grad = 2 * x  # The gradient of f(x) = x^2 is f'(x) = 2x

    is_correct, error = gradient_check(simple_quadratic, x, analytical_grad)
    print(f"Gradient check passed: {is_correct}, Relative error: {error:.2e}")