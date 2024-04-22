from typing import Tuple, Type
import math

import torch
import torch.nn.functional as F
from bitorch_engine.utils.quant_operators import nv_tensor_quant, gptq_stype_unpacking
from bitorch_engine.functions.cuda import tensor_to_packed_uint8, unpack_uint8_tensor


def flatten_x(x: torch.Tensor):
    """
    Flattens a 3D tensor into a 2D tensor by combining the first two dimensions.

    This is particularly useful for processing sequences in models like BERT/Transformers,
    where you might need to apply operations that expect 2D inputs.

    Args:
        x (torch.Tensor): A 3D tensor with shape [batch_size, seq_length, hidden_size].

    Returns:
        tuple[torch.Tensor, list]: A tuple containing the flattened 2D tensor with shape
        [batch_size * seq_length, hidden_size] and the original shape as a list
        [batch_size, seq_length] for later unflattening.
    """
    # shape of x in BERT/Transformer：[batch_size, seq_length, hidden_size]
    # flatten x to 2D tensor : [batch_size * seq_length, hidden_size]
    shape = list(x.size()[:-1])
    x = x.view(-1, x.size(-1))
    return x, shape


def unflatten_x(x: torch.Tensor, shape: list):
    """
    Unflattens a 2D tensor back into a 3D tensor using the original shape, reversing the operation
    performed by `flatten_x`.

    This function is useful for reconstructing the original 3D structure of sequence data
    after performing operations that require 2D input tensors.

    Args:
        x (torch.Tensor): A 2D tensor with shape [batch_size * seq_length, output_size].
        shape (list): The original shape of the tensor before flattening,
        as a list [batch_size, seq_length].

    Returns:
        torch.Tensor: The unflattened 3D tensor with shape [batch_size, seq_length, output_size].
    """
    # from [batch_size * seq_length, output_size] to [batch_size, seq_length, output_size]
    x = x.view(shape + [x.size(-1)])
    return x


## binary embedding layer helper functions
def pad_embedding_dim(weight: torch.Tensor) -> torch.Tensor:
    '''
    This function takes as input a PyTorch tensor "weight" representing the embedding matrix,
    and pads its embedding dimension to the smallest multiple of 8 that is greater than or
    equal to the current embedding dimension. It does so by calculating the remainder of the
    current embedding dimension divided by 8, and adding the required number of columns
    filled with -1 to the tensor. Finally, the function returns the padded tensor.

    Args:
        tensor (torch.Tensor): A PyTorch tensor for storing weight parameters

    Returns:
        tensor (torch.Tensor): Weight tensor after padding
    '''

    # Get the current embedding dimension
    curr_dim = weight.shape[1]

    # Check if the embedding dimension is a multiple of 8
    if curr_dim % 8 != 0:
        # Calculate the new padded embedding dimension
        new_dim = (curr_dim // 8 + 1) * 8
        # Calculate the number of columns to pad
        num_cols_to_pad = new_dim - curr_dim
        # Create the padding tensor filled with -1
        padding_tensor = torch.ones((weight.shape[0], num_cols_to_pad), dtype=torch.float, device=weight.device) * -1
        # Concatenate the padding tensor to the original tensor
        weight = torch.cat([weight, padding_tensor], dim=1)
    return weight


def pad_last_2_dims_to_multiple_of_128(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pad the last two dimensions of a PyTorch tensor to the nearest multiple of 128.

    This function ensures the last two dimensions of the input tensor are rounded up to the nearest multiple of 128,
    adding padding with zeros where necessary. This is particularly useful for preparing tensors for operations
    that require dimensions to be a certain size.

    Args:
        tensor (torch.Tensor): A PyTorch tensor that will be padded.

    Returns:
        torch.Tensor: The tensor padded to ensure its last two dimensions are multiples of 128.
        int: The number of elements added as padding to the second-to-last dimension.
    """
    # Get the original shape of the tensor
    padded_tensor = tensor
    orig_shape = tensor.shape
    # Compute the padded shape of the tensor by rounding up to the nearest multiple of 128
    padded_shape = list(orig_shape)
    pad_last = 0
    pad_sec_last = 0
    if padded_shape[-1] % 128 != 0:
        padded_shape[-1] = (padded_shape[-1] // 128 + 1) * 128
        pad_last = padded_shape[-1] - orig_shape[-1]
    if padded_shape[-2] % 128 != 0:
        padded_shape[-2] = (padded_shape[-2] // 128 + 1) * 128
        pad_sec_last = padded_shape[-2] - orig_shape[-2]
    padding = (0, pad_last, 0, pad_sec_last)
    if pad_last != 0 or pad_sec_last != 0:
        padded_tensor = F.pad(tensor, padding, mode='constant', value=0)

    return padded_tensor, pad_sec_last


def binary_matmul_forward_post_processing(tensor: torch.Tensor,
                                          shape_pre: list,
                                          x_pad_sec_last: int,
                                          y_pad_sec_last: int,
                                          k: int) -> torch.Tensor:
    """
    Post-processes the output tensor of a binary matrix multiplication operation.

    This function performs several post-processing steps on the result of a binary matrix multiplication,
    including truncating any padded elements added during the operation, reshaping the tensor back to its
    original dimensions with additional specified dimensions, and converting the binary data back to its
    original data domain.

    Args:
    - tensor (torch.Tensor): The output tensor from a binary matrix multiplication to be post-processed.
    - shape_pre (list): The original shape of the tensor before the binary matrix multiplication, which the output tensor will be reshaped to, with the last two dimensions replaced by the actual last two dimensions of the post-processed tensor.
    - x_pad_sec_last (int): The number of padded elements added to the second to last dimension of the tensor during the binary matrix multiplication. These will be removed.
    - y_pad_sec_last (int): The number of padded elements added to the last dimension of the tensor during the binary matrix multiplication. These will be removed.
    - k (int): A constant used to convert the binary data in the tensor back to its original data domain. The conversion formula applied is `k - 2 * tensor`.

    Returns:
    - torch.Tensor: The post-processed tensor, reshaped to its original dimensions with specified adjustments and converted back to its original data domain.

    Note:
    - This function is specifically designed for tensors resulting from binary matrix multiplication operations that involve padding and require post-processing to revert to their original format and domain.
    """
    ## truncate padded elements in m und n dim
    if x_pad_sec_last > 0:
        tensor = tensor[:, :-x_pad_sec_last, :]
    if y_pad_sec_last > 0:
        tensor = tensor[:, :, :-y_pad_sec_last]
    # reshape to (bs, num_head, seq_lengh, hid_size_per_head)
    tensor = tensor.view(shape_pre + [tensor.size(-2), tensor.size(-1)])
    # convert to (-1, 1) data domain
    tensor = k - 2 * tensor
    return tensor


def prepare_bie_layers(model: torch.nn.Module, layers=None) -> None:
    """
    Prepares binary and n-bit quantized layers within a given model for training or inference.
    This function iterates over the modules of the model and calls `prepare_params` on those
    which are instances of the specified quantized layer classes. This preparation step is
    essential for initializing or transforming parameters specific to quantized operations.

    Args:
        model (torch.nn.Module): The model containing the layers to be prepared.
        layers (list, optional): A list of layer classes to be prepared. If not provided,
            defaults to a predefined list of binary and n-bit quantized layer classes,
            including both convolutional and linear layers, as well as binary embedding layers.

    The function imports necessary classes from the `bitorch_engine` package, focusing on
    binary and n-bit implementations of convolutional layers, linear layers, and embedding layers.
    If no specific layers are provided, it defaults to a comprehensive list of available
    quantized layer types. Each layer in the model that matches a type in the `layers` list
    will have its `prepare_params` method called, allowing for any necessary parameter
    initialization or adjustments before the model is used.

    This is particularly useful for models that utilize quantized layers, ensuring that
    all such layers are correctly set up for either training or deployment.
    """

    # Import statements for binary and n-bit quantized layers from bitorch_engine
    from bitorch_engine.layers.qconv.binary import BinaryConv2dBase
    from bitorch_engine.layers.qconv.nbit import nBitConv2dBase
    from bitorch_engine.layers.qlinear.binary import BinaryLinearBase
    from bitorch_engine.layers.qlinear.nbit import nBitLinearBase, MPQLinearBase
    from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingCuda

    if not layers:
        layers = [BinaryConv2dBase, nBitConv2dBase, BinaryLinearBase, nBitLinearBase, MPQLinearBase, BinaryEmbeddingCuda]

    for idx, module in enumerate(model.modules()):
        if idx > 0:  # starts from the second item
            module_type = type(module)
            if any(issubclass(module_type, layer) for layer in layers) or any(isinstance(module_type, layer) for layer in layers):
                module.prepare_params()


def pack_bie_layers(model: torch.nn.Module, qweight_only: bool = True, layers=None) -> None:
    '''
    Packs the weights of quantization layers in a given model to prepare for efficient storage.
    This function should be invoked prior to using `torch.save()` for saving the model,
    ensuring that the quantized weights are properly compressed.

    Args:
        model: The model whose quantization layers' weights are to be packed. This model
                  should already be trained and contain quantization layers that support
                  weight packing.
        qweight_only: A boolean flag indicating whether only the weights are to be quantized
                         and packed. If `True`, only weights are packed, excluding other parameters
                         like biases. Defaults to `True`.
        layers: A list of layer classes that should be considered for packing. If not provided,
                   defaults to a predefined list of binary and n-bit quantized convolutional and
                   linear layer bases. This allows customization of which layers are to be packed
                   based on the model architecture.

    Note:
        The function iterates through all sub-modules of the provided model, checking if any
    module matches the types specified in the `layers` list. For each matching module, it calls
    the `generate_quantized_weight` method with the `qweight_only` parameter, which performs the
    actual weight packing process.
    '''
    from bitorch_engine.layers.qconv.binary import BinaryConv2dBase
    from bitorch_engine.layers.qconv.nbit import nBitConv2dBase
    from bitorch_engine.layers.qlinear.binary import BinaryLinearBase
    from bitorch_engine.layers.qlinear.nbit import nBitLinearBase, MPQLinearBase

    if not layers:
        layers = [BinaryConv2dBase, nBitConv2dBase, BinaryLinearBase, nBitLinearBase, MPQLinearBase]

    for idx, module in enumerate(model.modules()):
        if idx > 0:  # starts from the second item
            module_type = type(module)
            if any(issubclass(module_type, layer) for layer in layers):
                module.generate_quantized_weight(qweight_only=qweight_only)


def save_checkpoint(model: torch.nn.Module, name: str, qweight_only: bool = True) -> None:
    '''
    Saves the state of a quantized PyTorch model in a bit-packed format. This function is intended for models 
    that incorporate quantized layers, allowing for efficient storage and potential speedups in model loading 
    and inference.

    The function first packs the layers of the model based on the quantization status of the weights and then 
    saves the model's state dictionary. The saved checkpoint can be used for inference or to resume training, 
    depending on the inclusion of unpacked weights.

    Args:
        model (torch.nn.Module): The model to save. This model should use quantized layers.
        name (str): The file path where the model checkpoint will be saved. This path should include the
            filename and the desired file extension (usually ".pth" for PyTorch models).
        qweight_only (bool, optional): A flag to indicate whether to save only the quantized weights (True)
            or to also include the original, unpacked weights (False). Saving only quantized weights reduces file
            size but may limit the ability to resume training. Defaults to True, optimizing for reduced storage.

    Returns:
        None
    '''
    pack_bie_layers(model, qweight_only)
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, name)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, qweight_only: bool = True) -> None:
    """
    Loads a checkpoint into a given model. This function first applies weight packing to the model if
    quantized weights are used, then loads the model's state dict from the checkpoint path provided.
    This is particularly useful for models that use quantized weights, allowing the option to load
    only the quantized weights for inference or both quantized and unpacked weights for further training.

    Args:
        model: The model into which the checkpoint will be loaded. This model should use quantized layers if
            qweight_only is set to True.
        checkpoint_path: The file path to the checkpoint from which the model state will be loaded.
        qweight_only: A boolean flag indicating whether to pack and load only the quantized weights
            (True) or to also consider unpacked weights which can be useful for resuming
            training (False). Default is True, which means only quantized weights are considered.
    """
    pack_bie_layers(model, qweight_only)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)


def init_weight(weight: torch.Tensor, cls: Type[torch.nn.Parameter]=torch.nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Initializes binary parameters using pre-trained weights if available.

    This function calculates the weight scale from either the provided pre-trained weights
    or the initial weights. It converts weights to int8 for training, achieving a 4x reduction
    in size, and prepares for a fully bit-packed uint8 conversion for inference, achieving
    a 32x reduction in size. The process aims to preserve the average magnitude of the original weights.

    Args:
        weight (Tensor): The initial floating-point weight tensor.
        cls (Type[torch.nn.Parameter]): class of the output weight.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the initialized weight as a torch.nn.Parameter in int8 format
        and the scale of the weight.
    '''

    # Calculate weight scale from the pre-trained weights if provided, else from the initial weight.
    # Converts the weight tensor to float if not already in that dtype to ensure consistency
    w_f = weight
    if w_f.dtype != torch.float:
        w_f = w_f.to(torch.float)

    # Calculate the scale of the weights based on their L1 norm, divided by the number of elements.
    # This captures the average magnitude, which may better represent asymmetrically distributed weights.
    scale_w = w_f.norm(p=1).div(w_f.nelement()).to(weight.device)

    # Center the weights around zero by subtracting the mean. This step is crucial for the quantization process.
    weight = w_f - w_f.mean()

    # Quantize weights to int8 using a custom quantization function (assumed to be nv_tensor_quant here).
    # Replace zeros with the sign of the original weight to maintain the sign information after quantization.
    weight_int8 = nv_tensor_quant(weight)[0]
    weight_int8 = torch.where(weight_int8==0, weight.sign(), weight_int8)

    # Convert the quantized weights into a torch.nn.Parameter of type int8 for further training or inference use.
    weight = cls(
        weight_int8.to(torch.int8)
    )

    return weight, scale_w


def qweight_update_fn(qweight: torch.nn.Parameter, exp_avg_s: torch.Tensor=None, exp_avg_l: torch.Tensor=None,
                       step: torch.Tensor=None, lr:float=1e-4, weight_decay:float=0.0, beta1:float=0.99,
                      beta2:float=0.9999, eps: float = 1e-6, dtype=torch.half, correct_bias=None, projector=None,
                      grad:torch.Tensor=None) -> None:
    """
    This method defines how to update quantized weights with quantized gradients.
    It may involve operations such as applying momentum or adjusting weights based on some optimization algorithm.

    Args:
        qweight (torch.nn.Parameter): The current quantized weight parameter to be updated.
        exp_avg_s (torch.Tensor, optional): Exponential moving average of squared gradients. Used in optimization algorithms like Adam.
        exp_avg_l (torch.Tensor, optional): Exponential moving average of the gradients. Also used in optimizers like Adam.
        step (torch.Tensor, optional): The current step or iteration in the optimization process. Can be used to adjust learning rate or for other conditional operations in the update process.
        lr (float, optional): Learning rate. A hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function.
        weight_decay (float, optional): Weight decay (L2 penalty). A regularization term that helps to prevent overfitting by penalizing large weights.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. A hyperparameter for optimizers like Adam.
        beta2 (float, optional): The exponential decay rate for the second-moment estimates. Another hyperparameter for Adam-like optimizers.
        eps (float, optional): A small constant for numerical stability.
        dtype (torch.dtype, optional): The data type to be used for computations.
        correct_bias (optional): Whether to apply bias correction (specific to certain models like BERT).
        projector (optinal): Whether use a gradient projector.
        grad (optional): gradient tensor will be used if projector used.

    Returns:
        None: The function is expected to update the `qweight` in-place and does not return anything.

    Raises:
        NotImplementedError: Indicates that the function has not yet been implemented.
    """
    # import corresponding q-layers
    from bitorch_engine.layers.qembedding.binary import BinaryEmbeddingParameter
    from bitorch_engine.layers.qlinear.binary import BinaryLinearParameter
    from bitorch_engine.layers.qlinear.nbit import nBitLinearParameter, MPQWeightParameter
    from bitorch_engine.layers.qconv.binary import BinaryConvParameter
    from bitorch_engine.layers.qconv.nbit import nBitConvParameter
    from bitorch_engine.layers.qlinear.nbit.cuda.utils import pack_fp_weight

    # update step
    step.add_(1)

    # for binary embedding layers
    if isinstance(qweight, BinaryEmbeddingParameter):

        # for packed binary embedding weight
        if qweight.data.dtype is torch.uint8:
            scale_w = torch.ones((qweight[0], 1), dtype=torch.float)
            # unpack uint8 weight to float tensor (-1, 1) then mul lr
            v = unpack_uint8_tensor(qweight.grad, scale_w).mul_(lr).to(dtype)
        # for unpacked binary embedding weight
        elif qweight.data.dtype is torch.bool:
            v = qweight.grad.to(dtype)
            # (0, 1) to (-1, 1) and mul lr
            v = torch.where(v == 0, torch.tensor(-1, dtype=dtype, device=qweight.device), v).mul_(lr)
        else:
            raise NotImplementedError("qweight.dtype '{}' has not been supported yet.".format(str(qweight.data.dtype)))

        exp_avg_s.lerp_(v, (1 - beta2))

        if qweight.data.dtype is torch.uint8:
            binary_grad = tensor_to_packed_uint8(exp_avg_s)
            assert (binary_grad.dtype == torch.uint8), \
                'binary embedding grad has incorrect dtype {}.'.format(str(binary_grad.dtype))
        else:
            # Generate a new bool tensor where each item is True if the corresponding item in exp_avg_s is >= 0, otherwise False
            binary_grad = exp_avg_s >= 0

        # represents the items involved in update
        active_indices = qweight.active_indices
        # Use bitwise XOR to find differing bits, since XOR returns True if the bits are different
        differing_bits = qweight[active_indices] ^ binary_grad[active_indices]
        # Modify qweight where the bits are different
        qweight[active_indices] ^= differing_bits

    #  binary conv and linear layers.
    elif isinstance(qweight, (BinaryLinearParameter, BinaryConvParameter)):
        exp_avg_l.lerp_(qweight.grad.to(dtype), (1 - beta1))
        v = exp_avg_l.clone().sign_().mul_(lr)
        exp_avg_s.lerp_(v, (1 - beta2))
        u = exp_avg_s.clone().sign_().mul_(-1)
        u[u == 0] = 1
        mask = (u != qweight.sign())
        # sign flipping for binary weight update
        qweight.data.copy_(torch.where(mask, -qweight.data, qweight.data))
        # === debug stuff === #
        # flips = mask.view(-1).sum()
        # print(flips)
        # =================== #

    # q4 or q8 layers
    elif isinstance(qweight, (nBitLinearParameter, nBitConvParameter)):
        # int8 to floating-point dtype
        grad = qweight.grad.to(dtype)
        w = qweight.data.to(dtype)

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg_l.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
        exp_avg_s.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = exp_avg_s.sqrt().add_(eps)

        step_size = lr
        if correct_bias:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** step.item()
            bias_correction2 = 1.0 - beta2 ** step.item()
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        w.addcdiv_(exp_avg_l, denom, value=-step_size)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if weight_decay > 0.0:
            w.add_(w, alpha=(-lr * weight_decay))

        # update int8 qweight
        qweight.data = nv_tensor_quant(w)[0]

    elif isinstance(qweight, MPQWeightParameter):

        # unpack qweight
        w = gptq_stype_unpacking(qweight).to(dtype)

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg_l.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
        exp_avg_s.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = exp_avg_s.sqrt().add_(eps)

        step_size = lr
        if correct_bias:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** step.item()
            bias_correction2 = 1.0 - beta2 ** step.item()
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        # compute norm gradient
        norm_grad = exp_avg_l / denom

        # GaLore Projection Back
        if projector is not None:
            norm_grad = projector.project_back(norm_grad.to(dtype))

        w.add_(norm_grad, alpha=-step_size)

        if weight_decay > 0.0:
            w.add_(w, alpha=(-lr * weight_decay))

        # pack fp weight back to Q-weight and update qweight data
        qweight.data = pack_fp_weight(w, qweight)

        # manually empty cuda cache.
        del w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError("qweight.dtype '{}' has not been supported yet.".format(str(qweight.data.dtype)))
