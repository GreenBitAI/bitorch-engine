import platform
from enum import Enum
import subprocess
import os

class ARCH_CPU(Enum):
    '''
    Indicates which CPU architecture using for computation
    '''
    ARM_A76 = 1
    ARM_A55 = 2


class linux_arch_ident():
    """
    A utility class for identifying Linux architecture, specifically aimed at ARM architectures.

    This class provides methods to check if the current system is running on an ARM architecture
    and to determine the specific model of the ARM CPU.
    """
    @staticmethod
    def is_arm() -> bool:
        """
        Determines if the current system's architecture is ARM-based.

        :return: True if the system is ARM-based, False otherwise.
        """
        return platform.machine().lower().startswith('arm') or platform.machine().lower().startswith('aarch64')

    @staticmethod
    def get_arm_model() -> ARCH_CPU:
        """
        Fetches the model name of the ARM CPU from the system and maps it to a predefined ARCH_CPU enum.

        This method attempts to execute the 'lscpu' command to retrieve CPU information, then parses
        the output to identify the model name of the ARM CPU.

        :return: An ARCH_CPU enum value corresponding to the ARM CPU model.
        :raises Exception: If there's an error executing 'lscpu', or if the CPU model cannot be recognized.
        """
        try:
            # Execute 'lscpu' command and decode the output to get CPU information
            cpuinfo = (subprocess.check_output("lscpu", shell=True).strip()).decode().lower()
            # Parse the output to find the model name
            s = cpuinfo.index("model name:")
            s += len("model name:")
            n = cpuinfo[s:].index("\n")
            model_n = cpuinfo[s:s+n].strip()
        except:
            # Raise an exception if 'lscpu' command fails or if parsing fails
            raise Exception("Error occurred while running 'lscpu', please check if your OS supports this command.")

        # Map the model name to a specific ARCH_CPU enum value
        if model_n.__contains__("cortex-a55"):
            return ARCH_CPU.ARM_A55
        elif model_n.__contains__("cortex-a76"):
            return ARCH_CPU.ARM_A76
        # Raise an exception if the model name does not match known values
        raise Exception("Invalid architecture name obtained: {}.".format(model_n))


def check_cpu_instruction_support(search_term):
    """
    Checks if the CPU supports a specific instruction set.

    This function utilizes the `cpuinfo` library to fetch detailed CPU information,
    then searches for a specific term within the CPU flags to determine if a given
    CPU instruction set is supported.

    Args:
        search_term (str): The CPU instruction set or feature to search for, e.g., "sse4_2", "avx2".

    Returns:
        bool: True if the search term is found within the CPU flags, indicating support for the instruction set.
              False otherwise.

    Example:
        >>> check_cpu_instruction_support("sse4_2")
        True  # Indicates that "sse4_2" is supported by the CPU.

    Note:
        The function prints the entire CPU information fetched by `cpuinfo.get_cpu_info()` which can be quite verbose.
        Consider commenting out the `print` statement for production use.
    """
    import cpuinfo
    print(cpuinfo.get_cpu_info())
    # Check if the search term is present in the CPU flags
    if search_term in cpuinfo.get_cpu_info()["flags"]:
        return True
    else:
        return False
