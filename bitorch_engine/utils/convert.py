from typing import Iterable, Callable, Type, Optional

import torch
from torch import nn


def collect_layers(module, layer_types=None, parent_name=""):
    """
    Collect the names of all layers within a module recursively with certain types.

    :param module: the model
    :param layer_types: the desired layer types
    :param parent_name: the name of the parent (usually empty when called directly)
    :return: a list of names of all layers with the desired types
    """
    if layer_types is None:
        layer_types = [nn.Linear]

    if type(module) in layer_types:
        return {parent_name: module}

    result = {}
    for name, child in module.named_children():
        result.update(
            collect_layers(
                child,
                layer_types=layer_types,
                parent_name=parent_name + "." + name if parent_name != "" else name,
            )
        )
    return result


def replace_layers(
    module: torch.nn.Module,
    names_to_replace: Iterable[str],
    class_: Type,
    replace_fn: Callable[[torch.nn.Module], torch.nn.Module],
    parent_name: str = "",
):
    """
    This function replaces all layers (recursively) within the given modules, whose names are included in the given
    list. It requires a function replace_fn that constructs the replacement object for each given layer. Creating
    a list of layers can be done with the `collect_layers` function.

    :param module: the (sub-)network module, in which the layers should be replaced
    :param names_to_replace: the names of all layers to be replaced
    :param class_: the replacement class
    :param replace_fn: function which creates an instance of the replacement class
    :param parent_name: the name of the parent (usually empty when called directly)
    :return:
    """
    # TODO: check and support replacement in sequential
    if isinstance(module, class_):
        return []

    replaced_layers = []

    for attr in dir(module):
        tmp = getattr(module, attr)
        full_name = parent_name + "." + attr if parent_name != "" else attr
        if full_name in names_to_replace:
            setattr(
                module,
                attr,
                replace_fn(tmp),
            )
            assert isinstance(
                getattr(module, attr), class_
            ), "The replacement function does not create an object of the correct class. Recursion could fail/loop."
            replaced_layers.append(getattr(module, attr))

    # recursively call on children
    for child_name, child in module.named_children():
        replaced_layers.extend(
            replace_layers(
                child,
                names_to_replace,
                class_,
                replace_fn,
                parent_name + "." + child_name if parent_name != "" else child_name,
            )
        )
    return replaced_layers


def _get_simple_linear_replace_fn(class_):
    def replace_fn(previous_layer):
        return class_(previous_layer.in_features, previous_layer.out_features)

    return replace_fn


def get_mpq_config(mpq_strategy: Optional[str] = None):
    """
    This function returns the parameters for a given strategy string. Default is "2-8-32".

    Currently, known options are "2-8-32", "2-32-32", "2-128-32", "4-128-256", "8-128-256" (weight bits, group size, double quantized group size).
    :param mpq_strategy: the strategy string
    :return: args needed to be set for the MPQ layer
    """
    if mpq_strategy is None:
        mpq_strategy = "2-32-32"
    base_config = {"dq_mode": 2, "use_gba_quant": True, "asym": False}
    config_dict = {
        "2-8-32": {"w_bit": 2, "group_size": 8, "dq_group_size": 32},
        "2-32-32": {"w_bit": 2, "group_size": 32, "dq_group_size": 32},
        "2-128-32": {"w_bit": 2, "group_size": 128, "dq_group_size": 32},
        "4-128-256": {"w_bit": 4, "group_size": 128, "dq_group_size": 256},
        "8-128-256": {"w_bit": 8, "group_size": 128, "dq_group_size": 256},
        # asymmetric currently not supported
        # "2-32-asym": {"w_bit": 2, "group_size": 32, "dq_group_size": -1, "asym": True},
        # "4-32-256-asym": {"w_bit": 4, "group_size": 32, "dq_group_size": 256, "asym": True},
    }
    assert mpq_strategy in config_dict.keys(), f"{mpq_strategy} unknown!"
    base_config.update(config_dict[mpq_strategy])
    return base_config


def quantize_linear_with_mpq_linear_cuda(
    module: torch.nn.Module,
    names_to_replace: Iterable[str],
    mpq_strategy: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    parent_name: str = "",
):
    """
    Replace all layers contained in `names_to_replace` within the given module with `MPQLinearCuda` layers.
    :param module: the module which contains Linear layers
    :param names_to_replace: the list of layer names to be replaced
    :param mpq_strategy: which MPQ strategy to use, see `get_mpq_config`
    :param dtype: the dtype of the new module
    :param parent_name: the name of the parent (usually empty when called directly)
    :return: the list of names of layers which were replaced
    """
    from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda

    mpq_quantization_config = get_mpq_config(mpq_strategy)

    def replace_fn(previous_layer):
        return MPQLinearCuda(
            in_channels=previous_layer.in_features,
            out_channels=previous_layer.out_features,
            dtype=dtype,
            **mpq_quantization_config,
        )

    return replace_layers(
        module, names_to_replace, MPQLinearCuda, replace_fn, parent_name
    )


def quantize_linear_with_q4_linear_cutlass(
    module: torch.nn.Module,
    names_to_replace: Iterable[str],
    parent_name: str = "",
):
    """
    Replace all layers contained in `names_to_replace` within the given module with `Q4LinearCutlass` layers.
    :param module: the module which contains Linear layers
    :param names_to_replace: the list of layer names to be replaced
    :param parent_name: the name of the parent (usually empty when called directly)
    :return: the list of names of layers which were replaced
    """
    from bitorch_engine.layers.qlinear.nbit.cutlass import Q4LinearCutlass

    return replace_layers(
        module,
        names_to_replace,
        Q4LinearCutlass,
        _get_simple_linear_replace_fn(Q4LinearCutlass),
        parent_name,
    )


def quantize_linear_with_binary_linear_cuda(
    module: torch.nn.Module,
    names_to_replace: Iterable[str],
    parent_name: str = "",
):
    """
    Replace all layers contained in `names_to_replace` within the given module with `BinaryLinearCuda` layers.
    :param module: the module which contains Linear layers
    :param names_to_replace: the list of layer names to be replaced
    :param parent_name: the name of the parent (usually empty when called directly)
    :return: the list of names of layers which were replaced
    """
    from bitorch_engine.layers.qlinear.binary.cuda import BinaryLinearCuda

    return replace_layers(
        module,
        names_to_replace,
        BinaryLinearCuda,
        _get_simple_linear_replace_fn(BinaryLinearCuda),
        parent_name,
    )
