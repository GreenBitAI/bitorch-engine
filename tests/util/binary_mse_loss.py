import torch

def binary_mse_loss(output, target):
    print("Output:", output)
    print("Target:", target)
    loss = torch.mean((output.to(torch.float32) - target.to(torch.float32))**2)
    return loss
