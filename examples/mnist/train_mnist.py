"""
An example script for training a model for the MNIST dataset with Bitorch Engine.

Modified from the `PyTorch MNIST Example <https://github.com/pytorch/examples/blob/main/mnist/main.py>`_,
which was published under the `BSD 3-Clause License <https://github.com/pytorch/examples/blob/main/LICENSE>`_.
"""
# fmt: off
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from datasets import MNIST

from bitorch_engine.layers.qlinear.binary.cuda import BinaryLinearCuda
from bitorch_engine.layers.qlinear.nbit.cutlass import Q8LinearCutlass, Q4LinearCutlass
from bitorch_engine.optim import DiodeMix
from bitorch_engine.utils.model_helper import prepare_bie_layers

class BinaryMLP(nn.Module):
    def __init__(self, num_hidden_units_1: int=1024, num_hidden_units_2: int=1024, bits: int=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, num_hidden_units_1)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(num_hidden_units_1)

        # defines binary linear layer
        if bits == 1:
            self.fc2 = BinaryLinearCuda(num_hidden_units_1, num_hidden_units_2)
        elif bits == 4:
            self.fc2 = Q4LinearCutlass(num_hidden_units_1, num_hidden_units_2)
        else:
            self.fc2 = Q8LinearCutlass(num_hidden_units_1, num_hidden_units_2)

        self.act2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(num_hidden_units_2)

        self.fc3 = nn.Linear(num_hidden_units_2, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn2(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, choices=["q_mlp", "lenet"], default="lenet",
                        help='input batch size for training (default: 128)')
    parser.add_argument('--scheduler', type=str, choices=["step", "cosine"], default="cosine",
                        help='scheduler for training (default: cosine)')
    parser.add_argument('--bits', type=int, choices=[1, 4, 8], default="1",
                        help='quantization bit-width for q_mlp (default: 1-bit)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N',
                        help='num of workers for dataloader (default: 2)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset, test_dataset = MNIST.get_train_and_test("datasets/",download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # 1-bit the optimizer needs smaller learning rate
    lr = 1e-5 if args.bits == 1 else args.lr

    if args.model == "q_mlp":
        model = BinaryMLP(bits=args.bits).to(device)

        # prepares parameters of BIE layers
        prepare_bie_layers(model)
    else:
        model = Net().to(device)

    # defines Diode optimizer which is an optimizer specifically designed for BNN
    optimizer = DiodeMix(model.parameters(), lr=lr, dtype=torch.float)

    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise NotImplementedError('scheduler: {} not implemented yet!'.format(args.scheduler))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


    if args.save_model:
        torch.save(model.state_dict(), "mnist_example_bnn.pt")


if __name__ == '__main__':
    main()
