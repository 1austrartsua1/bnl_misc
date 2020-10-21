#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
import os


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def example(rank, world_size, nodeid, processes_per_node):

    localrank = rank
    del rank
    globalrank = nodeid*processes_per_node + localrank

    print(f"globalrank = {globalrank}")
    print(f"world_size={world_size}")
    print(f"nodeid={nodeid}")
    print(f"processes_per_node={processes_per_node}")

    dist.init_process_group("gloo", rank=globalrank, world_size=world_size)

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Example")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 1)",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    args.batch_size = args.batch_size // world_size

    print("Global Rank", globalrank, "World size", world_size, "Batch size", args.batch_size)

    kwargs = {"num_workers": args.workers, "pin_memory": True}

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=globalrank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **kwargs
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    run_results = []
    for _ in range(args.n_runs):

        print("Trying to make model on globalrank", globalrank)

        model = DDP(models.resnet18(num_classes=10).to(globalrank), device_ids=[globalrank])

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        for epoch in range(1, args.epochs + 1):
            train(args, model, globalrank, train_loader, optimizer, epoch)
        run_results.append(test(args, model, globalrank, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"resnet_{args.lr}_"
        f"{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


def main():
    processes_per_node = int(os.environ.get('PROCESSES_PER_NODE'))
    numNodes = int(os.environ.get('SLURM_NNODES'))
    nodeid = int(os.environ.get('SLURM_NODEID'))
    world_size = numNodes*processes_per_node
    
    # print(f"processes_per_node={processes_per_node}")

    if world_size == None:
        print("Error: missing world size")
    mp.spawn(example,
             args=(world_size,nodeid,processes_per_node,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
