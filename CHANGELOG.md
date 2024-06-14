# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.2.6] - 2024/06/14

### Added

- Installation instructions for binary releases
- Warning if non-customized PyTorch version is detected which can not calculate gradients for non-complex tensor types

### Changed

- Updated development scripts for binary releases
  - Adjusting rpaths in .so files (based on PyTorch's implemented solution)
  - Docker base image changed to manywheel builder image

## [0.2.5] - 2024/05/24

### Added

- Development scripts for preparing binary releases

### Changed

- Updated build instructions to clarify torchvision installation
- Adapted `setup.py` logic for preparing binary releases

### Fixed

- Broken build process by setting setuptools version

## [0.2.4] - 2024/05/23

### Added

- Tuned the hyperparameters of DiodeMix optimizer for sft.
- Added sft-support for the classical gptq-style models.
- Implemented qzeros update in finetuning process.

### Updated

- Extended pack_fp_weight function.
- Enhanced the performance of MPQLinearCUDA layer.

### Fixed

- Fixed various errors in DiodeMix update function.

## [0.2.3] - 2024/05/01

### Updated

- Enhanced the performance of the MBWQ linear layer for processing long sequences, addressing previous inefficiencies.

## [0.2.2] - 2024/04/29

### Updated

- Building instructions (adding a section for cutlass)
- Checksums for custom torch builds (within docker)

### Fixed

- An error in `pack_fp_weight`

## [0.2.1] - 2024/04/27

### Fixed

- Broken links in README.md and index.rst

## [0.2.0] - 2024/03/10

### Added

- Quantized layers with different acceleration options
  - QConv (binary, quantized) - CPU, Cutlass
  - QLinear (binary, quantized, mixed bit-width) - CUDA, Cutlass, MPS
  - QEmbedding (binary)
- Optimizer(s) for quantized layers
  - Hybrid optimizer `diode_beta` based on Diode v1 (binary) and AdamW (quantized) for memory-efficient training
  - Initial support for galore projection
- Examples
  - MNIST training script with and without PyTorch Lightning

## [0.1.0] - 2023/01/13

The first release of basic functionality.
