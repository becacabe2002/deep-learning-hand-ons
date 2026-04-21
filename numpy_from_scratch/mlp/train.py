import numpy as np
from model import *
from shared.data_utils import load_mnist_numpy, NumpyDataLoader
from shared.optimizers import Adam, SGD 
from numpy_from_scratch.linear_nn.model import LinearLayer
import time

def train_mlp_mnist(train_size=10000, batch_size=32,activation_func="relu", optimizer="adam", epochs=50):
    (X_train, y_train), (X_test, y_test) = load_mnist_numpy(n_samples=train_size, flatten=True)

    train_loader = NumpyDataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = NumpyDataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    #define a model with 28 * 28 = 784 input features, 128 hidden nodes and 10 output classes

    layer1 = LinearLayer(784, 128)
    if activation_func == "relu":
        act_func = ReLu()
    elif activation_func == "lrelu":
        act_func = LeakyReLu()
    else:
        raise ValueError("activation_func must be \"relu\" or \"lrelu\".")
        act_func = None
    layer2 = LinearLayer(128, 10)

    model = MLP([layer1, act_func, layer2])
    loss_fn = SoftMaxCrossEntropyLoss()

    params = [layer1.weights, layer1.bias, layer2.weights, layer2.bias]
    if optimizer == "adam":
        opt = Adam(params=params)
    elif optimizer == "sgd":
        opt = SGD(params=params)
    else:
        opt = None
        raise ValueError("optimizer must be \"adam\" or \"sgd\".")
    print(f"Training a simple MLP on MNIST dataset for {epochs} epochs")
    print(f"Input samples: {train_size}, activation function: {activation_func}, optimizer: {optimizer}.")

    start_time = time.perf_counter()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            # forward
            logits = model.forward(batch_X)
            loss = loss_fn.forward(logits, batch_y)

            # backward
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)

            # update params 
            grads = [layer1.dw, layer1.db, layer2.dw, layer2.db]
            opt.step(grads)

            total_loss += loss * len(batch_X)
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == batch_y)
            total_samples += len(batch_X)

        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}- Accuracy: {acc:.4f}")
    
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    print(f"Training time (s): {exec_time:4f}")

    # final eval
    test_correct = 0
    for batch_X, batch_y in test_loader:
        logits = model.forward(batch_X)
        preds = np.argmax(logits, axis=1)
        test_correct += np.sum(preds == batch_y)
    
    test_acc = test_correct / len(y_test)

    print(f"\n Accuracy on test dataset: {test_acc:.4f}")

if __name__ == "__main__":
    train_mlp_mnist(activation_func="relu", epochs=20)





