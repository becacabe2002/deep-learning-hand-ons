import os 
import urllib.request
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import torchvision as tv
from sklearn.datasets import make_swiss_roll, make_moons
import mnist

class NumpyDataLoader:
    """
    A lightweight data loader for npdarrays, mimicking the behavior of PyTorch's DataLoader API.
    """
    pass

    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.n_samples = X.shape[0]

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
def get_swiss_roll_2d(n_samples=1000, noise=0.1) -> np.ndarray:
    """
    Generate a 2D Swiss Roll dataset using sklearn's make_swiss_roll function.
    The Swiss Roll is a popular synthetic dataset used for testing dimensionality reduction techniques.
    It consists of points arranged in a spiral shape, making it ideal for visualizing the performance of algorithms like PCA, t-SNE, and UMAP.
    """
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    X_2d = X[:, [0, 2]]  # Take only the x and z coordinates for 2D visualization
    # Standardize the data to prevent issues with scale during training
    X_2d = (X_2d - X_2d.mean(axis=0)) / X_2d.std(axis=0)
    return X_2d.astype(np.float32)

def get_moons_2d(n_samples=1000, noise=0.1):
    """
    Generate a 2D Moons dataset using sklearn's make_moons function.
    The Moons dataset consists of two interleaving half circles, making it a popular choice for testing clustering and classification algorithms.
    The noise parameter adds Gaussian noise to the data, making the classification task more challenging.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X.astype(np.float32), y.astype(np.int64)

def load_mnist_numpy(n_samples=None, flatten=False):
    """
    Loads MNIST dataset as NumPy arrays, then normalized to range [0, 1]"""
    train_images = mnist.train_images().astype(np.float32) / 255
    train_labels = mnist.train_labels().astype(np.int64)
    test_images = mnist.test_images().astype(np.float32) / 255
    test_labels = mnist.test_labels().astype(np.int64)

    if n_samples is not None:
        train_images = train_images[:n_samples]
        train_labels = train_labels[:n_samples]
        test_images = test_images[:n_samples]
        test_labels = test_labels[:n_samples]
    
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    return (train_images, train_labels), (test_images, test_labels)

def load_mnist_pytorch(batch_size=64, train=True, n_samples=None, root='./data') -> DataLoader:
    """
    Loads MNIST dataset using PyTorch's torchvision, then normalized to range [0, 1]"""
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.1307,), (0.3081,)) # Normalize with mean and std of MNIST dataset
    ])
    dataset = tv.datasets.MNIST(root=root, train=train, download=True, transform=transform)

    if n_samples is not None:
        dataset = Subset(dataset, range(min(n_samples, len(dataset)))) # safeguard

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def download_tiny_shakespeare(root='./data')->str:
    """
    Downloads the Tiny Shakespeare dataset, a small text corpus often used for character-level language modeling tasks.
    The dataset contains the complete works of William Shakespeare, making it a rich source of text for training and testing language models.
    """
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, 'tiny_shakespeare.txt')
    if not os.path.exists(file_path):
        print(f"Downloading Tiny Shakespeare dataset to {file_path}...")
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            print(f"Error occurred while downloading the dataset: {e}")
    else:
        print(f"Tiny Shakespeare dataset already exists at {file_path}.")
    return file_path

def load_tiny_shakespeare(root='./data')->str:
    """
    Loads the Tiny Shakespeare dataset from the specified root directory.
    If the dataset is not already present, it will be downloaded first.
    """
    file_path = download_tiny_shakespeare(root)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text