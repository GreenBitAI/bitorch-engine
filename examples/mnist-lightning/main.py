from __future__ import print_function

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.fabric.strategies import DDPStrategy, DataParallelStrategy
from lightning_fabric.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from mlp import MLP


def get_default_supported_precision(training: bool, tpu: bool = False) -> str:
    """Return default precision that is supported by the hardware.

    Args:
        training: `-mixed` or `-true` version of the precision to use
        tpu: whether TPU device is used

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    if tpu:
        return "32-true"
    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return "bf16-mixed" if training else "bf16-true"
    return "16-mixed" if training else "16-true"


def train(model, train_loader, fabric, optimizer, epoch, dry_run, log_interval, tracing=False):
    fabric.print("Training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True) if tracing else nullcontext():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss, retain_graph=True)
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset) / fabric.world_size,
                    100. * batch_idx / len(train_loader), loss.item()))
                if dry_run:
                    break


def test(model, test_loader, fabric):
    fabric.print("Validating...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += len(data)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, 100. * correct / total)
    )


def quantize_model(fabric, model, quantize_method, dtype):
    from bitorch_engine.utils.convert import collect_layers

    layers_to_quantize = collect_layers(model)
    method_found = False
    replaced_layers = []

    do_not_quantize = ["first_linear", "last_linear"]
    for layer in do_not_quantize:
        if layer in layers_to_quantize:
            del layers_to_quantize[layer]

    if quantize_method.startswith("mpq"):
        method_found = True
        from bitorch_engine.utils.convert import quantize_linear_with_mpq_linear_cuda

        mpq_strategy = quantize_method.replace("mpq", "")
        if mpq_strategy == "":
            mpq_strategy = None

        replaced_layers = quantize_linear_with_mpq_linear_cuda(model, layers_to_quantize, mpq_strategy=mpq_strategy, dtype=dtype)

    if quantize_method == "q4_cutlass":
        method_found = True
        from bitorch_engine.utils.convert import quantize_linear_with_q4_linear_cutlass
        replaced_layers = quantize_linear_with_q4_linear_cutlass(model, layers_to_quantize)

    if quantize_method == "binary":
        method_found = True
        from bitorch_engine.utils.convert import quantize_linear_with_binary_linear_cuda
        replaced_layers = quantize_linear_with_binary_linear_cuda(model, layers_to_quantize)

    if not method_found:
        raise RuntimeError(f"Quantization method {quantize_method} unknown or not implemented yet.")

    assert len(replaced_layers) == len(layers_to_quantize), f"Some layers apparently were not successfully quantized! (To quantize: {layers_to_quantize}, but replaced only {replaced_layers}.)"
    return replaced_layers


def main(
        batch_size: int = 64,
        test_batch_size: int = 128,
        epochs: int = 20,
        lr: Optional[float] = None,
        no_cuda: bool = False,
        no_mps: bool = False,
        dry_run: bool = False,
        seed: int = 1,
        log_interval: int = 50,
        save_model: bool = False,
        precision: Optional[str] = None,
        tpu: bool = False,
        quantize: Optional[str] = None,
        debug_port: int = -1,
        resume: Union[bool, Path] = False,
) -> None:
    quantize_options = [None, "binary", "q4_cutlass", "mpq"]
    assert quantize in quantize_options, f"quantize option must be one of: {quantize_options}, was ({quantize})"
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    if quantize and tpu:
        raise RuntimeError("TPU and Quantization not yet supported.")
    print("Using precision '{}'".format(precision))

    # set default lr
    if lr is None:
        lr = 1e-4
        if quantize == "binary":
            lr = 1e-5
        if quantize is None:
            lr = 0.1

    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if debug_port > 0:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=debug_port, stdoutToServer=True, stderrToServer=True)

    accelerator = "auto"
    if use_cuda:
        accelerator = "cuda"
    elif use_mps:
        accelerator = "mps"

    logger = CSVLogger("logs", name="my_exp_name")
    wandb_logger = WandbLogger()

    # TODO: test and investigate possible bugs with (D)DPStrategy
    # strategy = DDPStrategy()
    # strategy = DataParallelStrategy()
    # fabric = L.Fabric(strategy=strategy, accelerator=accelerator, precision=precision, loggers=[logger, wandb_logger])
    fabric = L.Fabric(strategy="auto", accelerator=accelerator, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(f"Training strategy: {fabric.strategy}")
    fabric.launch()

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MLP()
    if quantize:
        dtype = torch.bfloat16 if "bf16" in precision else torch.float16
        quantized_layers = quantize_model(fabric, model, quantize, dtype)
        fabric.print(f"Quantized layers: {quantized_layers}")
    fabric.print(f"Model: {model}")

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    model = fabric.setup(model)
    if quantize:
        from bitorch_engine.utils.model_helper import prepare_bie_layers
        prepare_bie_layers(model)

    if quantize:
        # use the Diode optimizer for (partially) quantized networks
        from bitorch_engine.optim import DiodeMix
        optimizer = DiodeMix(model.parameters(), lr=lr, dtype=torch.float)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    fabric.print(f"Using optimizer {optimizer.__class__.__name__} and scheduler {scheduler.__class__.__name__}.")
    optimizer = fabric.setup_optimizers(optimizer)

    for epoch in range(1, epochs + 1):
        fabric.print(f"Starting epoch {epoch} with learning rate(s): {scheduler.get_lr()}.")
        train(model, train_loader, fabric, optimizer, epoch, dry_run, log_interval, tracing=(debug_port > 0))
        test(model, test_loader, fabric)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
