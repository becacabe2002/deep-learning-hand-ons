# deep-learning-hand-ons

Structured repository containing hands-on implementations of deep learning models. The project is divided into two main tracks:
- **NumPy from Scratch:** Focusing on the fundamental math and backpropagation.
- **Modern Frameworks (PyTorch):** Focusing on architectural design and efficient data pipelines.

## Implementation Roadmap

### 1. Infrastructure & Shared Utilities
- [ ] Configure `pyproject.toml` and run `uv sync` to install dependencies
- [ ] Create and parameterize the `Makefile` for generic training commands
- [ ] Implement `shared/optimizers.py` (SGD, Momentum, Adam for NumPy)
- [ ] Implement `shared/data_utils.py` (Data loaders for MNIST, Tiny Shakespeare, etc.)

### 2. NumPy from Scratch (Fundamental Math)
- [ ] `00_gradient_checking`: Numerical gradient checking (finite differences)
- [ ] `01_linear_nn`: Linear and Logistic Regression
- [ ] `02_mlp`: Multi-Layer Perceptrons / Feedforward Networks

### 3. Modern Frameworks (PyTorch)
- [ ] `01_cnn_foundations`: Modern CNNs (ResNet blocks, spatial pooling)
- [ ] `02_rnn_foundations`: Sequence modeling (GRUs/LSTMs)
- [ ] `03_vision_transformer`: ViT from scratch
- [ ] `04_diffusion`: Minimal DDPM (2D Swiss Roll & MNIST)
- [ ] `05_micro_gpt`: Causal language modeling (Tiny Shakespeare, RoPE, KV-caching)

## Usage

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

To get started, sync the environment:

```bash
uv sync
```

Use the `Makefile` to run training modules (once implemented):

```bash
uv run make train DIR=numpy_from_scratch.02_mlp
```

## Project Structure

```text
deep-learning-hand-ons/
├─ shared/                <-- Centralized utilities
├─ numpy_from_scratch/    <-- Pure NumPy implementations
└─ frameworks_modern/     <-- PyTorch implementations
```
