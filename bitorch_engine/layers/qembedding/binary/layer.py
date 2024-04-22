import typing
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math
from bitorch_engine.utils.model_helper import pad_embedding_dim, qweight_update_fn
import torch.nn.functional as F

if torch.cuda.is_available():
    from bitorch_engine.functions.cuda import tensor_to_packed_uint8, unpack_uint8_tensor


class BinaryEmbeddingParameter(Parameter):
    """
    A custom parameter class for binary embeddings, extending torch.nn.Parameter.

    This class is designed to support binary embedding layers, particularly useful
    in models requiring efficient memory usage and specialized optimization techniques.
    It extends the standard PyTorch Parameter by allowing the specification of active
    indices, which can be used to perform sparse updates only on certain parts of the
    parameter tensor, enhancing performance and efficiency during training.

    Attributes:
        active_indices (torch.Tensor): A tensor specifying the indices of the parameter
                                       that should be considered for updates. This allows
                                       for sparse updates, focusing on parts of the parameter
                                       that are actively used in the current context.

    Args:
        data (torch.Tensor, optional): The initial data for the parameter. Defaults to None.
        requires_grad (bool, optional): Flag indicating whether gradients should be computed
                                        for this parameter in the backward pass. Defaults to True.
        active_indices (torch.Tensor, optional): Specifies the active indices for sparse updates.
                                                 Defaults to None.
    """
    def __new__(cls,
                data: torch.Tensor=None,
                requires_grad: bool=True,
                active_indices: torch.Tensor=None,
                ):
        cls.active_indices = active_indices
        return super().__new__(cls, data=data, requires_grad=requires_grad)

    @staticmethod
    def update(qweight: torch.nn.Parameter, exp_avg_s: torch.Tensor=None, exp_avg_l: torch.Tensor=None,
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
        # Apply XOR operation directly on qweight to update, only for indexes in active_indices
        # bool_grad_weight (torch.Tensor): Binarized gradient, Boolean or uint8 type.
        # active_indices (torch.Tensor): The actual embedded index used by the current batch.
        # We usually direct use input (batch_size, seq_length).
        assert isinstance(qweight, BinaryEmbeddingParameter), 'Error: the type of qweight must be ' \
                                                              'BinaryEmbeddingParameter. '
        qweight_update_fn(qweight=qweight, exp_avg_s=exp_avg_s, exp_avg_l=exp_avg_l,
                          step=step, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                          correct_bias=correct_bias, eps=eps, dtype=dtype, projector=projector, grad=grad)


class BinaryEmbeddingForward(Function):
    """
    Experimental class for forward pass of binary embeddings.
    Note: This class is experimental and may not always work as expected.
    Due to the use of uint8 type for weight parameters (qweight), which are bit-packed binary weights,
    a custom optimizer is necessary for training.

    Args:
        input (torch.Tensor): Input tensor with shape (batch_size, seq_length).
        weight (torch.Tensor): The embedding weight matrix, bit-packed.
        embed_scale (torch.Tensor): Scaling factor for the embeddings.
        ori_embedding_dim (int): Original embedding dimension before packing.
        is_train (bool): Flag indicating if the operation is in training mode.

    Returns:
        torch.Tensor: The resulting tensor after applying embedding lookup and unpacking.
    """
    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                qweight: torch.Tensor,
                embed_scale: torch.Tensor,
                ori_embedding_dim: int,
                is_train: bool) -> torch.Tensor:
        """
        Forward pass for binary embedding lookup.

        This function performs embedding lookup for each index in the input tensor,
        using bit-packed binary weights. It supports dynamic scaling and unpacking
        of the weights to retrieve the original embeddings.

        Args:
            ctx: Context object for backward pass.
            input (torch.Tensor): Input tensor containing indices, with shape (batch_size, seq_length).
            qweight (torch.Tensor): Bit-packed binary weights tensor for embeddings.
            embed_scale (torch.Tensor): Scaling factors for the embeddings, to be applied after unpacking.
            ori_embedding_dim (int): Original embedding dimension before packing.
            is_train (bool): Flag indicating whether the operation is in training mode.

        Returns:
            torch.Tensor: The resulting embedding tensor after lookup and scaling, with shape
            (batch_size, seq_length, original_embedding_dim).

        Note:
            This function saves necessary tensors for backward pass if in training mode.
        """
        # input shape (batch_size, seq_length)
        # emd has shape (batch_size, seq_length, packed_embedding_dim)
        # scl has shape (batch_size, seq_length, 1)
        emd = qweight.index_select(dim=0, index=input.flatten()).view(input.size(0), input.size(-1), -1)
        scl = embed_scale.index_select(dim=0, index=input.flatten()).view(input.size(0), input.size(-1), -1)

        if is_train:
            ctx.save_for_backward(input, qweight, scl)

        # r has shape (batch_size, seq_length, packed_embedding_dim * 8)
        r = unpack_uint8_tensor(emd, scl).to(embed_scale.dtype)
        # convert back to the original embedding_dim if padded
        if r.size(-1) > ori_embedding_dim:
            num_cols_to_truncate = r.size(-1) - ori_embedding_dim
            r = r[:, :, :-num_cols_to_truncate]
        return r

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Backward pass for the BinaryEmbeddingForward function, providing gradients for the input and packed weights.

        This method is designed to handle the gradient flow for bit-packed parameters, specifically for binary embeddings,
        ensuring the correct update behavior in conjunction with a custom optimizer designed for such data types.

        Args:
            ctx (torch.autograd.function.BackwardCFunction): The context object where saved tensors are retrieved.
            output_gradient (torch.Tensor): The gradient of the loss with respect to the output of the forward pass.

        Returns:
            typing.Tuple: A tuple containing gradients for the inputs used in the forward pass. Most elements are `None`
                          as only specific parameters require gradients.

        Note:
            The optimizer update behavior suggested for this setup involves a specialized approach. Here is an example
            of how to perform the necessary operations:

            - Perform XOR operation: apply XOR between qweight and grad_weight
            - Create a mask identifying non-zero positions in grad_weight
            - Update qweight only at positions where grad_weight is non-zero, keeping the original qweight values elsewhere

            This process is essential due to the binary and packed nature of the weights, requiring careful manipulation
            to ensure correct updates during training.

            "sparse_update_embedding_qweight" method can be used for qweight-update in an optimizer.
        """
        input, qweight, embed_scale = ctx.saved_tensors

        # Gradient for the input is not computed as it's not required for the embedding layer.
        grad_input = None

        # Compute gradients for the embedding weight matrix, adjusting for the packed and scaled format.
        # Initialize a tensor for unpacked gradients with the correct shape and type.
        # The grad_output tensor has shape (batch_size, seq_length, embedding_dim)
        w_unpacked = torch.zeros((qweight.size(0), qweight.size(1) * 8), dtype=output_gradient.dtype,
                            device=output_gradient.device)

        # Adjust gradients if necessary to match the unpacked weight shape.
        if output_gradient.size(-1) < w_unpacked.size(-1):
            pad_size = [0, w_unpacked.size(-1) - output_gradient.size(-1)]
            output_gradient = F.pad(output_gradient, pad_size, mode='constant', value=-1)

        # Apply scaling to the output gradient before accumulation.
        output_gradient.mul_(embed_scale.expand_as(output_gradient))

        # Accumulate gradients into the unpacked weight tensor.
        w_unpacked.index_add_(dim=0,
                              index=input.view(-1),
                              source=output_gradient.view(-1, w_unpacked.size(dim=1))
                              )

        # The grad_weight tensor has shape (vocab_size, embedding_dim / 8)
        # w_unpacked binarized and bit-packed into uint8 tensor.
        grad_weight = tensor_to_packed_uint8(w_unpacked)

        # active_indices: only operate on the indexes contained in active_indices.
        # In this way, we ensure that only the weights related to the current input are updated,
        # consistent with the sparsity of the weight update pattern of the Embedding layer.
        # "sparse_update_embedding_qweight" method can be used for update qweight in optimizer
        qweight.active_indices = input

        return grad_input, grad_weight, None, None, None


class BinaryEmbeddingCuda(nn.Module):
    """
    Binarized version of embedding layer, currently in experimental stage.
    Note: This class is experimental and may not always work as expected.
    It supports uint8 type weight parameters (qweight) that are bit-packed, requiring a custom optimizer for training.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (Optional[int]): Specifies a padding index. Embeddings at this index will be initialized with zeros.
        dtype (torch.dtype): Data type of the embeddings.

    Attributes:
        weight (Parameter): The original floating-point weights, not used during forward passes but necessary for initializing qweight.
        qweight (Parameter): The quantized, bit-packed weights used for embeddings.
        scale_w (torch.Tensor): Row-wise scaling factor for the embedding dictionary.
    """
    def __init__(
        self,
        *args: int,
        num_embeddings: int, # dict size
        embedding_dim: int,  # vector length
        padding_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float,
        **kwargs: int,
    ) -> None:
        super().__init__()
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx

        self.padding_idx = padding_idx
        self.dtype = dtype
        self.embedding_dim = embedding_dim # vector size
        self.num_embeddings = num_embeddings # dictionary size
        self.init_weight()

    def init_weight(self) -> None:
        """
        Initializes weight parameters. This includes the floating-point weights (for initial setup),
        the quantized and bit-packed weights (qweight), and the scaling factor (scale_w).
        """
        self.weight = Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim)),
            requires_grad=False
        )
        nn.init.normal_(self.weight)

        w_shape = pad_embedding_dim(self.weight).shape
        self.qweight = BinaryEmbeddingParameter(
            torch.zeros((w_shape[0], math.ceil(w_shape[1]/8)), dtype=torch.uint8)
        )

        # row-wise scaling factor for the embbeding dictionary, will be set while initializing weight
        self.register_buffer(
            'scale_w',
            torch.zeros((w_shape[0], 1), dtype=self.dtype)
        )

    def prepare_params(self) -> None:
        """
        Prepares and initializes the binary weight (qweight) and scale factor (scale_w) for the embedding.
        Must be called after model initialization and checkpoint loading.

        One can use "prepare_bie_layers" method from project_root.utils.model_helper to call this function.
        """
        # self.qweight has not been modified or has not been loaded from a checkpoint.
        # we will initialize qweight and scale_w
        if torch.all(self.qweight == 0) or torch.all(self.scale_w == 0):
            weight = self.weight - self.weight.mean()
            self.reset_parameters(weight)

            # pad embedding_dim if required
            # pads its embedding dimension to the smallest multiple of 8 that is greater than
            # or equal to the current embedding dimension.
            weight = pad_embedding_dim(weight)

            # bitpacking uses unsigned int as container
            w = tensor_to_packed_uint8(weight)

            assert (w.dtype == torch.uint8), \
                'binary embedding weight has incorrect dtype {}.'.format(str(w.dtype))
            assert (w.nelement() == weight.nelement()/8), \
                'The size of packed weight should be "{}", but got "{}".'\
                    .format(weight.nelement()/8, w.nelement())

            # initialize weight-scaling
            w_f = weight
            if weight.dtype != torch.float:
                w_f = w_f.to(torch.float)
            self.scale_w = w_f.norm(1, 1, keepdim=True)\
                              .div(w_f[0].nelement())\
                              .to(w_f.device)

            # NOTE:
            # Random initialize the weight with torch.uint8 type
            # two special changes need to be done if we want to train this layer:
            # 1. a special pytorch version supporting non-float back-propagation
            # 2. a special optimizer supporting torch.uint8 weight and grad_weight update.
            self.qweight = BinaryEmbeddingParameter(
                w,
                requires_grad=True if self.training else False
            )
        else:
            # qweight has been loaded successfully
            pass

        # release
        del self.weight
        # Manually trigger PyTorch's garbage collector
        import gc
        gc.collect()
        torch.cuda.empty_cache()


    def reset_parameters(self, weight: torch.Tensor) -> None:
        """
        Resets parameters, including filling the padding index with zeros if specified.

        Args:
            weight (torch.Tensor): The weight tensor to reset.
        """
        self._fill_padding_idx_with_zero(weight)

    def _fill_padding_idx_with_zero(self, weight: torch.Tensor) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass to generate embeddings for input indices.

        Args:
            input (Tensor): Tensor of indices for which embeddings are to be generated.

        Returns:
            Tensor: The resulting embeddings.
        """
        return BinaryEmbeddingForward.apply(input, self.qweight, self.scale_w, self.embedding_dim, self.training)


class BinaryEmbeddingBagForward(Function):
    """
    An experimental PyTorch function for forward pass of binary embedding bag.

    This class represents a custom autograd function for a binary embedding bag,
    designed to work with boolean weight parameters. Specialized optimizers are
    required for training due to the boolean nature of weights and their gradients.

    The forward pass performs an embedding lookup and binarizes the output based
    on the majority of ones in the sliced embeddings.

    Note: This class is experimental and may not be error-free or always functional.

    Args:
        input (Tensor): Input tensor containing indices for embedding lookup.
        weight (Tensor): Boolean weight tensor for embeddings.
        is_train (bool): Flag indicating if the forward pass is for training.

    Returns:
        Tensor: The result tensor after applying binary embedding logic.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, is_train: bool):
        """
        The forward pass performs an embedding lookup and binarizes the output based
        on the majority of ones in the sliced embeddings.

        Note: This class is experimental and may not be error-free or always functional.

        Args:
            input (Tensor): Input tensor containing indices for embedding lookup.
            weight (Tensor): Boolean weight tensor for embeddings.
            is_train (bool): Flag indicating if the forward pass is for training.

        Returns:
            Tensor: The result tensor after applying binary embedding logic.
        """
        if is_train:
            ctx.save_for_backward(input, weight)
        # Lookup the embedding for each index in the input tensor
        # The input tensor has shape (batch_size, seq_length)
        # The output tensor has shape (batch_size, embedding_dim)
        bag_of_words = weight.index_select(dim=0, index=input.flatten()).view(input.shape[0], -1, weight.size(dim=1))
        ones = torch.count_nonzero(bag_of_words, dim=1)
        r = ones.ge(math.ceil(input.size(1)/2)).to(input.dtype)
        # Convert to binary representation (1., -1.)
        r = torch.where(r == 0, torch.tensor(-1, dtype=input.dtype, device=input.device), r)
        return r

    @staticmethod
    @typing.no_type_check
    def backward(ctx: torch.autograd.function.BackwardCFunction,
                 output_gradient: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        """
        Implements the backward pass for the binary embedding bag function.

        This method simply passes the output gradient unchanged as the input gradient,
        which is a placeholder for future implementations of gradient calculations
        for boolean weights.

        Note on Optimizer Requirements for Boolean Weights:

        When both weights and their gradients are of boolean type, the optimizer must employ a specialized
        update mechanism. Traditional gradient descent methods cannot be directly applied since boolean
        values do not support the typical arithmetic operations involved in weight updates. Instead, the
        optimizer should implement logic that decides the binary state of weights based on certain criteria
        or rules derived from the boolean gradients. This might involve strategies like flipping the state
        of a weight based on the presence or absence of a gradient, or using a voting system across multiple
        training steps to determine the change. The development of such an optimizer requires careful
        consideration to effectively train models with binary weights while adhering to the limitations
        and characteristics of boolean algebra.
        "sparse_update_embedding_qweight" method can be used for qweight-update in an optimizer.

        Args:
            ctx (Any): Autograd context saving input and weight tensors for backward computation.
            output_gradient (torch.Tensor): Gradient of the loss with respect to the output of the forward pass.

        Returns:
            Tuple[None, torch.Tensor, None]: A tuple containing gradients for each input argument. Currently,
            only the gradient with respect to the weight tensor is calculated.
        """
        input, qweight = ctx.saved_tensors
        grad_input = None
        # Calculate the gradient of the loss with respect to the embedding weight matrix
        # The grad_output tensor has shape (batch_size, embedding_dim)
        # The grad_weight tensor has shape (vocab_size, embedding_dim)
        grad_weight = torch.zeros_like(qweight, dtype=output_gradient.dtype)
        grad_weight = grad_weight.index_add_(0, input.view(-1), output_gradient.repeat(input.shape[1], 1))

        # Directly binarize the gradient and convert it to Boolean type
        bool_grad_weight = torch.where(grad_weight >= 0, torch.tensor(True, device=output_gradient.device),
                                       torch.tensor(False, device=output_gradient.device))

        # active_indices: only operate on the indexes contained in active_indices.
        # In this way, we ensure that only the weights related to the current input are updated,
        # consistent with the sparsity of the weight update pattern of the EmbeddingBag layer.
        # "sparse_update_embedding_qweight" method can be used for update qweight in optimizer
        qweight.active_indices = input

        return grad_input, bool_grad_weight, None


class BinaryEmbeddingBag(nn.Module):
    """
    An binary embedding bag implementation.

    This module implements a binarized version of the standard embedding layer,
    utilizing boolean weights for embeddings. It is specifically designed for scenarios
    requiring binary weight parameters and includes an experimental optimizer for training.

    Note:
        This module is EXPERIMENTAL and not guaranteed to be error-free or always operational.
    Training requires a custom optimizer due to the boolean nature of weight parameters and gradients.

    Note on Boolean Weight Representation in PyTorch:

    PyTorch represents boolean (bool) type tensors using the Char type, which occupies 8 bits per value.
    Thus, despite being boolean in nature, the weights in this implementation are not truly 1-bit weights,
    as each boolean value is stored in an 8-bit format. This is important to consider when evaluating
    the memory efficiency and computational performance of models using these binary weights.

    The boolean weight.shape = (num_embeddings, embedding_dim), which is as same as the standard embedding layers,
    with 4x memory reduction by using Char tensor type.

    Note on Optimizer Requirements for Boolean Weights:

    When both weights and their gradients are of boolean type, the optimizer must employ a specialized
    update mechanism. Traditional gradient descent methods cannot be directly applied since boolean
    values do not support the typical arithmetic operations involved in weight updates. Instead, the
    optimizer should implement logic that decides the binary state of weights based on certain criteria
    or rules derived from the boolean gradients. This might involve strategies like flipping the state
    of a weight based on the presence or absence of a gradient, or using a voting system across multiple
    training steps to determine the change. The development of such an optimizer requires careful
    consideration to effectively train models with binary weights while adhering to the limitations
    and characteristics of boolean algebra.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (Optional[int]): Specifies a padding index. Embeddings at this index will be zeroed out.
    """
    def __init__(
        self,
        *args: int,
        num_embeddings: int, # dict size
        embedding_dim: int,  # vector length
        padding_idx: Optional[int] = None,
        **kwargs: int,
    ) -> None:
        super().__init__()
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        # NOTE:
        # Random initialize the weight with torch.bool type
        # two special changes need to be done if we want to train this layer:
        # 1. a special pytorch version supporting non-float backpropagation
        # 2. a special optimizer supporting boolean weight and boolean grad_weight update.
        self.weight = BinaryEmbeddingParameter(
                        torch.rand((num_embeddings, embedding_dim)) > 0.5,
                        requires_grad=True
                     )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameters by zeroing out the padding index if specified."""
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        """Zeroes out the embedding at the padding index, if it's specified."""
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(False)

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the binary embedding bag for given input indices.

        Args:
            input (Tensor): Tensor of indices to fetch embeddings for.

        Returns:
            Tensor: The resulting tensor after applying binary embedding bag logic.
        """

        # It first gets the input size and shape, then flattens the input tensor.
        return BinaryEmbeddingBagForward.apply(input, self.weight, self.training)
