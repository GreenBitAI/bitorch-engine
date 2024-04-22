    
    .
    ├── ...
        ├── layers                      # Low-bit layers
        │   ├── qconv                   # quantized Convolotional Layer 
        │   ├── qembedding              # currently only 1-bit embedding layer supported
        │   ├── ... 
        │   └── qlinear                 
        │       ├── binary (1-bit)
        │       │   ├── cpp             # x86 CPU
        │       │   ├── cuda            # Nvidia GPU
        │       │   └── cutlass         # Nvidia GPU
        │       └── n-bit (2/4/8-bit)
        │           ├── mps             # Apple GPU
        │           ├── cuda            # Nvidia GPU, e.g., weight-only quantized LLMs
        │           └── cutlass         # Nvidia GPU, e.g., quantization aware training for both activation and weight
        └── optim
        │    └── DiodeMix               # dedicated optimizer for low-bit quantized model 
        ├── functions
        └── ...