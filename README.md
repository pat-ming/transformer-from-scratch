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
| Linear Layer | Yes | Yes | Yes |
| ReLU | Yes | Yes | Yes |
| Feed-Forward Network | Yes | Yes | Yes |
| Residual Connections | Yes | Yes | Yes |
| Positional Encoding (Sinusoidal) | Yes | Yes | N/A (no learnable params) |
| Embedding | Yes | Yes | Yes |
| Multi-Head Attention | Yes | Yes | Yes |
| Transformer Block (Pre-Norm) | Yes | Yes | Yes |
| Cross-Entropy Loss (with Softmax) | Yes | Yes | Yes |
| Full Decoder-Only Transformer | Yes | Yes | Yes |
| SGD Optimizer | Yes | — | — |

### Architecture

Decoder-only (GPT-style) transformer with pre-norm residual connections:

```
Input IDs → Embedding → Positional Encoding → [TransformerBlock x N] → LayerNorm → Linear → Logits
```

Each TransformerBlock:
```
x → LayerNorm → Multi-Head Attention → + x → LayerNorm → FFN → + residual
```

## Training

Trained on a subset of Moby Dick using the GPT-2 tokenizer (vocab size 50,257). Achieved a loss of **0.0009** on the training set, confirming the model can learn.

Hyperparameters:
- `d_model = 128`
- `num_heads = 8`
- `num_layers = 2`
- `seq_len = 32`
- `optimizer = SGD`

## Testing

Every component is verified with **numerical gradient checking** (central difference method) to confirm that the analytical backward pass matches the numerical approximation. Tests are located at the bottom of `main.py`.

```bash
python main.py
```

A successful run prints gradient match results for each component.

## Project Structure

```
transformer.scratch/
├── main.py                          # All component implementations, tests, and training
├── training dataset/mobydick.txt    # Training data
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy tokenizers matplotlib
```

## Requirements

- Python 3.10+
- NumPy
- tokenizers (HuggingFace, for GPT-2 BPE tokenizer)
- matplotlib (for loss plotting)

## Author

Patrick Ming
