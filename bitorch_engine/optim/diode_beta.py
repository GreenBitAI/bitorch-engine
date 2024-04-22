import math
import warnings
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.optim import Optimizer

from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingParameter
from bitorch_engine.layers.qlinear.binary import BinaryLinearParameter
from bitorch_engine.layers.qlinear.nbit import nBitLinearParameter, MPQWeightParameter
from bitorch_engine.layers.qconv.binary import BinaryConvParameter
from bitorch_engine.layers.qconv.nbit import nBitConvParameter

from packaging import version

from .galore_projector import GaLoreProjector


def check_pytorch_version(required_version):
    """
    Checks if the current PyTorch version meets the specified minimum version requirement.

    Args:
        required_version (str): The minimum version of PyTorch required.

    Raises:
        Exception: If the current PyTorch version is below the required minimum version.
    """
    # Get the current version of PyTorch
    current_version = torch.__version__

    # Compare the current version with the required minimum version
    if version.parse(current_version) < version.parse(required_version):
        raise Exception(f"Current PyTorch version {current_version} is below the required minimum version {required_version}.")


class DiodeMix(Optimizer):
    """
    DiodeMix is a custom optimizer designed for efficient optimization, leveraging
    adaptive learning rates and momentum. It is particularly suited for deep learning
    tasks involving parameters with binary, n-bit, or standard floating-point values.
    This implementation is based on "Diode: Reinventing Binary Neural Networks Training
    with Sign Descent Optimization" Guo, Nianhui et al., 2024.

    Attributes:
        params (Iterable[nn.parameter.Parameter]): Iterable of parameters to optimize
        lr (float, optional): Learning rate. Defaults to 1e-4.
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square. Defaults to (0.99, 0.9999).
        eps (float, optional): Term added to the denominator to improve numerical stability.
            Defaults to 1e-6.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
        correct_bias (bool, optional): Whether to correct bias in adaptive learning rate.
            Defaults to True.
        dtype (torch.dtype): data type will be used in qweight update computation

    Methods:
        __init__(): Initializes the optimizer with the given parameters and options.
        step(closure=None): Performs a single optimization step.

    Raises:
        ValueError: If any of the parameters (learning rate, betas, epsilon) are out of their expected range.

    Note:
        This optimizer includes checks to ensure the learning rate and beta values are within
        valid ranges, raising ValueError if not. It supports sparse gradients with a specific
        error message guiding the user towards using SparseAdam instead if needed.
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.99, 0.9999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        dtype: torch.dtype = torch.float
    ):
        check_pytorch_version("1.5.0")  # add_ with alpha

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        self.dtype = dtype
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
                Defaults to None.

        Returns:
            The loss from the closure call, if any.

        Note:
            This method updates the weights of the parameters based on the gradients. It handles
            different types of parameters (binary, n-bit, or floating-point) differently to optimize
            their weights efficiently. For binary and n-bit parameters, it updates quantized weights
            directly. For standard floating-point parameters, it applies Adam-like updates.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                w, grad, lr, wd, beta1, beta2, state, eps, correct_bias = p, p.grad, group['lr'], group['weight_decay'], \
                                                       *group['betas'], self.state[p], group["eps"], group["correct_bias"]

                if isinstance(p, MPQWeightParameter):
                    grad = p.privileged_grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                if "step" not in state:
                    state['step'] = torch.zeros(1)

                # init GaLore Projection
                projector = None
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    projector = state["projector"]
                    grad = projector.project(grad.to(self.dtype), state["step"].item())

                # exp_avg initialization
                if "exp_avg_s" not in state:
                    delta = torch.rand_like(w, dtype=self.dtype).mul_(1e-3)
                    if isinstance(p, BinaryEmbeddingParameter):
                        state['exp_avg_s'] = -(w.data.clone().sign_().to(self.dtype).mul_(delta))
                    elif isinstance(p, (BinaryLinearParameter, BinaryConvParameter)):
                        state['exp_avg_l'] = torch.zeros_like(w, dtype=self.dtype)
                        state['exp_avg_s'] = -(w.data.clone().sign_().to(self.dtype).mul_(delta))
                    else:
                        state['exp_avg_l'] = torch.zeros_like(grad, dtype=self.dtype)
                        state['exp_avg_s'] = torch.zeros_like(grad, dtype=self.dtype)

                # q-layers update
                if isinstance(p, (BinaryLinearParameter, BinaryConvParameter, BinaryEmbeddingParameter,
                                  nBitLinearParameter, nBitConvParameter, MPQWeightParameter)):
                    p_class = type(p)
                    p_class.update(qweight=p, exp_avg_s=state["exp_avg_s"], exp_avg_l=state["exp_avg_l"],
                       step=state["step"], lr=lr, weight_decay=wd, beta1=beta1, beta2=beta2, correct_bias=correct_bias,
                       eps=eps, dtype=self.dtype, projector=projector, grad=grad)
                else:
                    # print('use standard adamw')
                    exp_avg, exp_avg_sq, step = state["exp_avg_l"], state["exp_avg_s"], state["step"]
                    step.add_(1)
                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)

                    step_size = lr
                    if correct_bias:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** step.item()
                        bias_correction2 = 1.0 - beta2 ** step.item()
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    # compute norm gradient
                    norm_grad = exp_avg / denom

                    # GaLore Projection Back
                    if "rank" in group:
                        norm_grad = projector.project_back(norm_grad)

                    w.add_(norm_grad, alpha=-step_size)

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        w.add_(w, alpha=(-lr * group["weight_decay"]))

        return loss
