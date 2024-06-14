

def initialize():
    """
    This functions makes all custom layer implementations available in BITorch.
    """
    from .layers.qlinear import QLinearInf


import os
torch_int_gradients_support = None
if os.environ.get("BIE_SKIP_TORCH_CHECK", "false") == "false" and torch_int_gradients_support is None:
    import warnings
    torch_int_gradients_support = False
    try:
        import torch
        x = torch.nn.Parameter(torch.zeros((1,), dtype=torch.uint8), requires_grad=True)
        torch_int_gradients_support = True
    except RuntimeError as e:
        if "dtype" in str(e).lower() and "only" in str(e).lower():
            warnings.warn(
                "It seems a regular version of torch is installed.\n"
                "  Please install the custom torch with enabled gradient calculation for integer tensors.\n"
                "  Check the instructions at https://github.com/GreenBitAI/bitorch-engine for more information.")
        else:
            warnings.warn("There may be a problem with the currently installed version of torch:\n" + str(e))
    except ModuleNotFoundError as e:
        # if torch is not installed, we can not check
        pass
