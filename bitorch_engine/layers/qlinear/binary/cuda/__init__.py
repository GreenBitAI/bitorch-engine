import torch

from .bmm import BMM

if torch.cuda.is_available():
    from .layer import BinaryLinearCuda
