# Example for MNIST

In this example script we train a simple model for the MNIST dataset using [Bitorch Engine](https://github.com/GreenBitAI/bitorch-engine).

First the requirements for this example need to be installed:
```bash
pip install -r requirements.txt
```

Then you can run the following to train an MLP with 3 layers (one of which is a binary layer),
or add `--help` for more arguments:
```bash
python train_mnist.py --epochs 10 --model q_mlp --log-interval 100
```
