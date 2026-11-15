"""Hardware checks."""


def gpu_available() -> bool:
    """Return True if a CUDA GPU is visible to PyTorch."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()