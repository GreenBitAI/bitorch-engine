from .layer import BinaryLinearBase, BinaryLinearParameter
import torch.cuda


def get_best_binary_implementation():
    if torch.cuda.is_available():
        from .cuda import BinaryLinearCuda
        return BinaryLinearCuda
    else:
        from .cpp import BinaryLinearCPP
        return BinaryLinearCPP


BinaryLinear = get_best_binary_implementation()
