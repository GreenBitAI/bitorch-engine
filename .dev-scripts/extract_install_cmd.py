# take caution: everything is quite hardcoded here
# any changes to the readme could break this code
# run it from root directory: python extract_install_cmd.py path/to/custom/torch-xxx.whl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("custom_pytorch_path", help="Path to custom PyTorch wheel")
parser.add_argument("custom_bitorch_engine_path", help="Path to built bitorch engine wheel file")
args = parser.parse_args()

BLOCK_HEADER_START_BINARY = "### Binary Release"
BLOCK_HEADER_START_FROM_SOURCE = "#### Conda on Linux"
BLOCK_END = "##########"

with open("README.md") as infile:
    content = infile.readlines()

with open(".dev-scripts/basic_tests.sh") as infile:
    test_appendix = infile.readlines()


def write_file(filepath, main_content):
    with open(filepath, "w") as outfile:
        outfile.write(FILE_INTRO)
        outfile.writelines(main_content)
        outfile.writelines(test_appendix)


source_local_install_instructions = []
source_global_install_instructions = []
binary_local_install_instructions = []
binary_global_install_instructions = []

in_code_block = False
reading_instructions = False
insert_block_pause = False
instruction_type = ""

FILE_INTRO = """#!/usr/bin/env bash

trap exit INT
set -o errexit
set -o xtrace

"""
EXTRA_CONDA_INSTRUCTION = """# extra step for bash script (not required in a proper command line):
eval "$(conda shell.bash hook)"
"""


for line in content:
    if line.startswith("```"):
        in_code_block = not in_code_block
        continue
    if line.startswith(BLOCK_HEADER_START_FROM_SOURCE):
        reading_instructions = True
        instruction_type = "source-global"
        BLOCK_END = BLOCK_HEADER_START_FROM_SOURCE.split()[0]
        continue
    if line.startswith(BLOCK_HEADER_START_BINARY):
        reading_instructions = True
        instruction_type = "binary-global"
        BLOCK_END = BLOCK_HEADER_START_BINARY.split()[0]
        continue
    if line.startswith("<details><summary>"):
        if "source" in instruction_type:
            instruction_type = "source-local"
        if "binary" in instruction_type:
            instruction_type = "binary-local"
        continue
    if line.startswith("</details>"):
        if "source" in instruction_type:
            instruction_type = "source-both"
        if "binary" in instruction_type:
            instruction_type = "binary-both"
        continue
    if line.startswith(BLOCK_END):
        reading_instructions = False
        continue
    if not reading_instructions:
        continue
    if not in_code_block:
        insert_block_pause = True
        continue

    # deal with comments
    if line.startswith("# export CC="):
        line = line[2:]
    if line.startswith("#"):
        continue

    # replace some line contents and add some lines
    if "conda activate" in line:
        line = EXTRA_CONDA_INSTRUCTION + line
    if "export BITORCH_WORKSPACE" in line:
        line = line.replace("${HOME}", "$(pwd)")
    if line.startswith("pip install torch-"):
        line = "pip install {}\n".format(args.custom_pytorch_path)
    if line.startswith("pip install bitorch_engine"):
        line = "pip install {}\n".format(args.custom_bitorch_engine_path)

    # decide how to write line
    line_format = "{line}"
    if line.startswith("#"):
        line_format = "{line}"
    if insert_block_pause:
        insert_block_pause = False
        line_format = "\n" + line_format

    # write result line(s)
    if instruction_type == "source-global" or instruction_type == "source-both":
        source_global_install_instructions.append(line_format.format(line=line))
    if instruction_type == "source-local" or instruction_type == "source-both":
        source_local_install_instructions.append(line_format.format(line=line))
    if instruction_type == "binary-global" or instruction_type == "binary-both":
        binary_global_install_instructions.append(line_format.format(line=line))
    if instruction_type == "binary-local" or instruction_type == "binary-both":
        binary_local_install_instructions.append(line_format.format(line=line))

write_file(".dev-scripts/test_source_local_conda_install.sh", source_local_install_instructions)
write_file(".dev-scripts/test_source_global_conda_install.sh", source_global_install_instructions)
write_file(".dev-scripts/test_binary_local_conda_install.sh", binary_local_install_instructions)
write_file(".dev-scripts/test_binary_global_conda_install.sh", binary_global_install_instructions)

binary_local_cu118 = [line.replace("cu121", "cu118").replace("cuda-12.1.0", "cuda-11.8.0") for line in binary_local_install_instructions]
write_file(".dev-scripts/test_binary_local_conda_install_cu118.sh", binary_local_cu118)
binary_local_no_cuda = filter(lambda x: "nvidia/label/cuda-12.1.0" not in x, binary_local_install_instructions)
write_file(".dev-scripts/test_binary_local_conda_install_no_cuda.sh", binary_local_no_cuda)
