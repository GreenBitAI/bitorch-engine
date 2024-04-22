
Welcome to Bitorch Engine's documentation!
==========================================

Welcome to the documentation of Bitorch Engine (BIE): a cutting-edge computation library for neural networks that enhances PyTorch by integrating specialized layers and functions for Low-Bit quantized neural network operations. This is where you can find all the information you need about how to use BIE.

Building on the foundational strengths of Bitorch Engine, the technology has been employed in pioneering projects that push the boundaries of neural network training and inference. For instance,

- `green-bit-llm-trainer <https://github.com/GreenBitAI/green-bit-llm/tree/main/sft>`_: In this project, BIE represents a significant leap in the field of Large Language Model (LLM) fine-tuning. Unlike traditional approaches that either quantize a fully trained model or introduce a few additional trainable parameters for `LoRA <https://github.com/microsoft/LoRA>`_ style fine-tuning, this project innovates by directly fine-tuning the quantized parameters of LLMs. This paradigm shift allows for the full-scale quantization fine-tuning of LLMs, ensuring that the training process tightly integrates with the quantization schema from the outset.
- `green-bit-llm-inference <https://github.com/GreenBitAI/green-bit-llm/tree/main/inference>`_ also showcases the BIE's adeptness at supporting inference for models quantized from 4 to 2-bits without any significant loss in accuracy compared to the original 32 or 16-bits models. It stands as a testament to BIE's capability to maintain the delicate balance between model size, computational efficiency, and accuracy, addressing one of the key challenges in deploying sophisticated neural networks in resource-constrained environments.

All changes are tracked in the `changelog <https://github.com/GreenBitAI/bitorch-engine/blob/main/CHANGELOG.md>`_.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   installation
   build_options
   documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Enjoy exploring our documentation!
