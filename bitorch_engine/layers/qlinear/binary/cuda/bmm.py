from enum import Enum


class BMM(Enum):
    """
    Enumeration for selecting the Bit-Matrix-Multiplication (BMM) kernel to be used during operations.
    This allows for the choice of different underlying implementations based on the requirements or
    optimizations desired for specific hardware or computational constraints.

    Attributes:
        BSTC32: Software-based Tensor Core implementation. This option utilizes a software-level implementation
                to simulate tensor core operations, potentially offering more flexibility at the cost of raw performance.
        BTC32: Bit-Matrix-Multiplication using NVIDIA Tensor Cores. This leverages hardware tensor cores for
               accelerated computation, suitable for NVIDIA GPUs that support tensor core operations, offering
               high performance for matrix multiplications.
        ADAPTIVE: Automatically selects the best combination of kernel implementations based on the specific dimension
                  constraints of the inputs and weights. This option aims to optimize performance by considering
                  the characteristics of the computation and available hardware capabilities.

    The choice of kernel can significantly affect the performance and efficiency of operations that involve
    matrix multiplications, especially in deep learning models where such operations are prevalent.
    """
    BSTC32 = 1  # software based tensor core implementation
    BTC32 = 2  # Bit-Matrix-Multiplication using NVIDIA Tensor Cores
    ADAPTIVE = 3  # Chooses the best kernel based on input and weight dimensions