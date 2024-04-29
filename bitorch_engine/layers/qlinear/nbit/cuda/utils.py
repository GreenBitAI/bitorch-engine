import torch
from bitorch_engine.layers.qlinear.nbit import MPQWeightParameter


def unpack_qweight(qweight: MPQWeightParameter) -> torch.Tensor:
    """
    Reconstructs the fp16 weight tensor from the input quantized weight parameter.

    Parameters:
        qweight (MPQWeightParameter): The quantized weight parameter object containing all necessary quantization information.

    Returns:
        torch.Tensor: The reconstructed weight tensor in fp16 format.

    Raises:
        ValueError: If essential attributes are missing in the input qweight parameter.
        NotImplementedError: For quantization types that are not yet supported.

    Supported quantization styles:
        1. GPTQ style with g_index.
        2. GPTQ style without g_index.
        3. Mixed-bit quantization.
    """
    layer_type = getattr(qweight, 'layer_type', None)

    if layer_type is None:
        raise ValueError("Error: invalid attribute of qweight in 'unpack_qweight'.")

    # Process based on layer type
    if qweight.layer_type == 1: # GPTQ style (with or without g_index)
        wf = torch.tensor(list(range(0, 32, qweight.w_bit)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
        weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // qweight.w_bit, -1),
                       wf.unsqueeze(-1)).to(torch.int16 if qweight.w_bit == 8 else torch.int8).view(-1, qweight.size(-1))
        torch.bitwise_and(weight, (2 ** qweight.w_bit) - 1, out=weight)

        if qweight.asym:
            zeros_unpack = torch.bitwise_right_shift(torch.unsqueeze(qweight.zeros, 2).expand(-1, -1, 32 // qweight.w_bit),
                                 wf.unsqueeze(0)).to(torch.int16 if qweight.w_bit == 8 else torch.int8)
            torch.bitwise_and(zeros_unpack, (2 ** qweight.w_bit) - 1, out=zeros_unpack)
            zeros_unpack = zeros_unpack + 1
            zeros = zeros_unpack.reshape(-1, qweight.size(-1))

            weights = qweight.scales[qweight.g_idx.long()] * (weight - zeros[qweight.g_idx.long()])
        else:
            # 2. GPTQ style without g_index.
            if qweight.g_idx is None:
                scales = qweight.scales.unsqueeze(1).repeat(1, weight.size(0)//qweight.scales.size(0), 1).view(-1, qweight.scales.size(-1))
                zeros = qweight.zeros.unsqueeze(1).repeat(1, weight.size(0) // qweight.zeros.size(0), 1).view(-1, qweight.zeros.size(-1))
                weights = weight.mul(scales) - zeros
            else:
                weights = weight * qweight.scales[qweight.g_idx.long()] - qweight.zeros[qweight.g_idx.long()]

    elif qweight.layer_type == 2: # MBWQLinear layer
        weights = None
        try:
            from bitorch_engine.layers.qlinear.nbit.cuda import MBWQLinearCuda
            use_mbwq = True if qweight.q_group_map is not None else False
            if not use_mbwq:
                weights = MBWQLinearCuda.q42fp_weight(qweight.data, qweight.scales, qweight.zeros, qweight.group_size,
                                                      qweight.w_bit, qweight.q_perm)
            else:
                weights = MBWQLinearCuda.exl2fp_weight(qweight.data, qweight.scales, qweight.zeros, qweight.q_perm,
                                                       qweight.q_group_map, qweight.rows)
        except ModuleNotFoundError as e:
            print(f"Error: Module not found: {e}.")
    else:
        raise NotImplementedError("Error: 'layer_type' not yet supported!")

    return weights


def pack_fp_weight(weight: torch.Tensor, qweight: MPQWeightParameter) -> torch.Tensor:
    """Packs the fp16 weight into a quantized weight format using the attributes defined in the QweightParameter.

    This function handles three main scenarios:
        1. GPTQ style quantization with group index (g_index).
        2. GPTQ style quantization without g_index.
        3. Mixed-bit quantization (currently not implemented).

    Parameters:
        weight (torch.Tensor): The floating-point weights to be quantized and packed.
        qweight (MPQWeightParameter): An object containing quantization parameters.

    Returns:
        torch.Tensor: The packed integer tensor representing the quantized weights.

    Raises:
        ValueError: If 'layer_type' attribute is invalid or not present.
        NotImplementedError: For unimplemented quantization methods, like mixed-bit quantization.
    """
    layer_type = getattr(qweight, 'layer_type', None)
    scales = getattr(qweight, 'scales', None)
    zeros = getattr(qweight, 'zeros', None)
    w_bit = getattr(qweight, 'w_bit', None)
    asym = getattr(qweight, 'asym', None)
    g_idx = getattr(qweight, 'g_idx', None)

    if layer_type is None:
        raise ValueError("Error: invalid 'layer_type' attribute in 'unpack_qweight' method.")

    # Process based on layer_type and existence of q_perm for quantization
    if layer_type == 1 or (layer_type == 2 and qweight.q_group_map is None): # MPQLinear or MBWQLinear-q4
        if asym:
            intweight = torch.round(weight / scales[g_idx] + zeros[g_idx]).to(torch.int32).clamp(0, 2**w_bit-1)
        else:
            if g_idx is None:
                # Adjust scales and zeros for symmetric quantization without group index
                scales = scales.unsqueeze(1).repeat(1, weight.size(0)//scales.size(0), 1).view(-1, scales.size(-1))
                zeros = zeros.unsqueeze(1).repeat(1, weight.size(0) // zeros.size(0), 1).view(-1, zeros.size(-1))
                if hasattr(qweight, "q_perm") and qweight.q_perm is not None:
                    q_perm = qweight.q_perm.unsqueeze(1).repeat(1, weight.size(1)).long()
                    weight = torch.gather(weight, dim=0, index=q_perm)

                intweight = torch.round((weight + zeros) / scales).to(torch.int32).clamp(0, 2 ** w_bit - 1)
            else:
                # Calculate integer weights for symmetric quantization with group index
                # TODO: recalculate scales and zeros?
                intweight = torch.round((weight + zeros[g_idx]) / scales[g_idx]).to(torch.int32).clamp(0, 2**w_bit-1)

        # Perform parallel bitpacking
        wf = torch.tensor(list(range(0, 32, w_bit)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
        intweight = torch.sum(
            torch.bitwise_left_shift(
                intweight.reshape(-1, 32 // w_bit, intweight.size(-1)),
                wf.unsqueeze(-1)
            ),
            dim=1,
            dtype=torch.int32
        )
    else:
        # TODO: Placeholder for mixed-bit-width quantization method
        raise NotImplementedError("Error: pack_fp_weight for MBWQLinear using mixed-bit-width not supported yet.")

    return intweight.to(torch.int32)


def make_group_map(q_groups: torch.Tensor, num_qrows: int) -> torch.Tensor:
    """
    Creates a mapping of quantization groups for handling irregular group sizes in quantized models.

    This function generates a tensor representing the mapping of groups, where each group might have
    a different size due to the quantization process. The mapping is used to organize or access quantized
    weights or parameters based on their group assignment.

    Parameters:
        q_groups (torch.Tensor): A tensor containing information about the quantization groups. It is expected
         to hold pairs of values, where each pair consists of 'bits' and 'start index' for each group.
        num_qrows (int): The total number of quantization rows, representing the overall size of the quantization
         dimension.

    Returns:
        torch.Tensor: A tensor of short integers representing the group mapping. Each group is represented by
        its index followed by the inverse row index within the group.

    Example:
        Given q_groups tensor indicating group sizes and num_qrows indicating the total quantization rows,
        this function calculates the group mapping required for accessing or organizing the quantized parameters.
    """
    num_groups = q_groups.numel() // 2
    group_map = []

    for i in range(num_groups):
        bits = q_groups[i * 2]
        if i < num_groups - 1:
            qrows = q_groups[i * 2 + 3] - q_groups[i * 2 + 1]
        else:
            qrows = num_qrows - q_groups[i * 2 + 1]
        rows = qrows * 32 // bits

        for j in range(rows):
            group_map.append(i)
            group_map.append(rows - j)

    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)