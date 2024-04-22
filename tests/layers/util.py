import os
import math
from typing import Union, List

import torch


def activate_remote_pycharm_debug(port: int = 11004):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True)


def to_device(data: torch.Tensor, device: torch.device) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def get_cuda_test_device_id():
    return int(os.environ.get("BIE_DEVICE", "0"))


def get_cuda_test_device():
    return torch.device(f"cuda:{get_cuda_test_device_id()}")


def get_mps_test_device():
    return torch.device("mps")


def get_packed_info(channels, n_bits, bits_prop, bits_group_size):
    groups = 0
    rows = 0
    bits_channel = []
    for idx in range(len(bits_prop)):
        if idx < len(bits_prop) - 1:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = max(1, int(channels * (bits_prop[idx])) // minimal_channels) * minimal_channels
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]
        else:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = channels - sum(bits_channel)
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]

    return groups, rows


def get_q_groups(groups, n_bits, group_size, channels, bits_prop):
    qgroups = []
    bits_column_end_index = []

    for idx in range(len(bits_prop)):
        if idx < len(bits_prop) - 1:
            minimal_columns = list(group_size.values())[idx]
            columns_index = max(1, int(channels * (
                bits_prop[idx])) // minimal_columns) * minimal_columns  # TODO: determine the minimal bits columns
            if idx > 0:
                columns_index += bits_column_end_index[-1]
            bits_column_end_index.append(columns_index)
        else:
            bits_column_end_index.append(channels)

    for bits_idx, bits in enumerate(n_bits):
        if bits_idx == 0:
            rows_per_bit = bits_column_end_index[bits_idx]
        else:
            rows_per_bit = bits_column_end_index[bits_idx] - bits_column_end_index[bits_idx - 1]

        gs = group_size[str(bits)]
        groups_per_bit = rows_per_bit // gs

        for group in range(groups_per_bit):
            qgroups.append(bits)  # record bits per group
            qgroups.append(0)

    out_row = 0
    rem_rows = channels
    for i in range(groups):
        bits = qgroups[2 * i]
        gs = group_size[str(bits)]

        rows_per_group = min(gs, rem_rows)  # rows per group before packing
        wpqr = 32 / bits  # INT32 elements per group for packing
        qrows = math.ceil(rows_per_group / wpqr)  # rows per group after packing
        qgroups[2 * i + 1] = out_row  # record packed rows start idx per group

        out_row += qrows

    return qgroups


def pack_rows_4_pytorch(input_tensor, rows, columns):
    # Calculate the number of output columns and store 4 columns per 32 bits, so use columns * 4 / 32
    out_columns = columns * 4 // 32

    # Initialize the output tensor, the size is [rows, out_columns], the data type is uint32
    output_tensor = torch.zeros((rows, out_columns), dtype=torch.int64, device=input_tensor.device)

    # To simulate the behavior of the CUDA kernel, we need to perform the same operation for each output element
    # This is implemented using the broadcast and indexing functions of PyTorch
    for row in range(rows):
        for out_column in range(out_columns):
            packed = 0
            for i in range(8):
                # Simulate operations in the CUDA kernel
                x = input_tensor[row, out_column * 8 + i].item() - 1
                packed |= (x << (i * 4))
            output_tensor[row, out_column] = packed
    # Use bitwise operators to extract the unsigned lower 32 bits
    # Note: 0xFFFFFFFF is a 32-bit all-1 mask, which is used to extract the lower 32 bits
    output_tensor = (output_tensor & 0xFFFFFFFF).to(torch.int32)
    return output_tensor