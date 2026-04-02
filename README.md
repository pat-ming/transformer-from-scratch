# Transformer from Scratch

A from-scratch implementation of the Transformer architecture using **only NumPy** — no PyTorch, TensorFlow, or any deep learning framework. Every forward and backward pass is hand-derived and implemented manually to build a deep understanding of the math behind Transformers.

## Motivation

The goal of this project is to understand the Transformer at the lowest level by implementing every component — including backpropagation — from first principles. By relying only on NumPy for tensor operations, nothing is hidden behind autograd or framework abstractions.

## Implemented Components

All tensors follow the shape convention `(batch, seq_length, d_model)`.

| Component | Forward | Backward | Gradient Verified |
|---|---|---|---|
| Layer Normalization | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes |
| Scaled Dot-Product Self-Attention | Yes | Yes | Yes |

### Layer Normalization
Normalizes inputs across the feature dimension with learnable `gamma` and `bias` parameters. The backward pass computes gradients for `dx`, `dgamma`, and `dbias`.

### Softmax
Numerically stable softmax using the max-subtraction trick. Supports batched 4D tensors `(batch, heads, seq, keys)` for use inside attention.

### Scaled Dot-Product Self-Attention
Computes `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V` with optional masking. The backward pass produces gradients `dQ`, `dK`, and `dV`.

## Roadmap

- [ ] Multi-Head Attention
- [ ] Position-wise Feed-Forward Network
- [ ] Positional Encoding
- [ ] Residual Connections
- [ ] Full Encoder Block
- [ ] Full Decoder Block
- [ ] End-to-end Transformer
- [ ] Training loop with a toy task

## Testing

Every component is verified with **numerical gradient checking** (central difference method) to confirm that the analytical backward pass matches the numerical approximation. Tests are located at the bottom of `main.py`.

```bash
python main.py
```

A successful run prints gradient match results for each component.

## Project Structure

```
transformer.scratch/
├── main.py          # All component implementations and tests
├── tests.ipynb      # Jupyter notebook for experimentation
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy
```

## Requirements

- Python 3.10+
- NumPy

## Author

Patrick Ming
