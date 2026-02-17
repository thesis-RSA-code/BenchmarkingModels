import torch.nn as nn


def get_activation(activation_name: str):
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU()
    }
    if activation_name.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation_name}. Choose from {list(activations.keys())}")
    return activations[activation_name.lower()]
