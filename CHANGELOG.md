# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


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
