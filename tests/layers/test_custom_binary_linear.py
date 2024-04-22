import pytest
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Parameter


class LinearFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight, bias)
        # TODO: instead of converting to float and use torch's mm we should use binary mm instead
        output = input.float().mm(weight.t().float())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # TODO: instead of converting to float and use torch's mm we should use our binary mm instead
            grad_input = grad_output.mm(weight.float())
        if ctx.needs_input_grad[1]:
            # TODO: instead of converting to float and use torch's mm we should use our binary mm instead
            grad_weight = grad_output.t().mm(input.float())
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        print("Weight grads calculated: ", grad_weight)

        # manually need to convert to our required formulation
        def binarize_grad(x):
            # 1.0 = True
            # 0.0 = False
            return torch.where(x >= 0.0, 1.0, 0.0)

        # TODO: we could inline this later
        grad_input = binarize_grad(grad_input)
        grad_weight = binarize_grad(grad_weight)

        return grad_input, grad_weight, grad_bias


# Option 2: wrap in a function, to support default args and keyword args.
def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)


class TLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


def test_q_linear_with_binary_weights():
    torch.manual_seed(42)
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=11004, stdoutToServer=True, stderrToServer=True)

    batch_size = 10
    num_inputs = 32
    num_outputs = 64

    # option 1: set dtype
    # currently not possible, check uniform bounds fails:
    # layer = TstLinear(num_inputs, num_outputs, bias=False, dtype=torch.bool)

    # option 2: manually replace weight:
    # currently possible
    # but we have to
    layer = TLinear(num_inputs, num_outputs, bias=False)
    layer.weight = Parameter(torch.rand((num_outputs, num_inputs)) > 0.5, requires_grad=True)

    input = Parameter(torch.rand((batch_size, num_inputs)) > 0.5, requires_grad=True)

    result = layer(input)
    print(result)
    print("Result shape: ", result.size())

    print("Grad before backward: ", layer.weight.grad)
    mse_loss = nn.MSELoss()
    dummy_loss = mse_loss(result, torch.ones_like(result) * 10)
    dummy_loss.backward()
    print("Grad after backward: ", layer.weight.grad)
    # technically it works, but we only get boolean gradients (all true currently)
    # could be fixed with custom backward pass?
