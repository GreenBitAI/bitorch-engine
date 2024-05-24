# take caution: everything is quite hardcoded here
# any changes to the readme could break this code
# run it from root directory: python extract_install_cmd.py path/to/custom/torch-xxx.whl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("custom_pytorch_path", help="Path to custom PyTorch wheel")
args = parser.parse_args()

BLOCK_HEADER_START = "#### Conda on Linux"

with open("README.md") as infile:
    content = infile.readlines()

local_install_instructions = []
global_install_instructions = []

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
    if line.startswith(BLOCK_HEADER_START):
        reading_instructions = True
        instruction_type = "global"
        continue
    if line.startswith("<details><summary>"):
        instruction_type = "local"
        continue
    if line.startswith("</details>"):
        instruction_type = "both"
        continue
    if line.startswith(BLOCK_HEADER_START.split()[0]):
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

    # decide how to write line
    line_format = "{line}"
    if line.startswith("#"):
        line_format = "{line}"
    if insert_block_pause:
        insert_block_pause = False
        line_format = "\n" + line_format

    # write result line(s)
    if instruction_type == "global" or instruction_type == "both":
        global_install_instructions.append(line_format.format(line=line))
    if instruction_type == "local" or instruction_type == "both":
        local_install_instructions.append(line_format.format(line=line))

with open(".dev-scripts/test_local_conda_install.sh", "w") as outfile:
    outfile.write(FILE_INTRO)
    outfile.writelines(local_install_instructions)
with open(".dev-scripts/test_global_conda_install.sh", "w") as outfile:
    outfile.write(FILE_INTRO)
    outfile.writelines(global_install_instructions)
