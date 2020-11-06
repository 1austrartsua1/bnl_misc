#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
import os
import time

from randomDataLoader import MyRandomDataSet




def train(model, device, train_loader, optimizer, epoch):
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
    return np.mean(losses)


def test(model, device, test_loader):
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


def example(rank, world_size, nodeid, cmdlineArgs, train_dataset, test_dataset):
    commType = cmdlineArgs.comm_backend
    processes_per_node = cmdlineArgs.processes_per_node
    localrank = rank
    del rank
    globalrank = nodeid*processes_per_node + localrank
    if globalrank == 0:
        tstartexample = time.time()

    dist.init_process_group(commType, rank=globalrank, world_size=world_size)

    # Training settings

    device = torch.device(cmdlineArgs.device)

    if cmdlineArgs.scaling_type == "strong":
        # strong scaling: keep total batchsize constant by cutting down the amount of work per GPU
        # weak scaling: keep work done per GPU costant so that total batchsize grows as we add more GPUs.
        cmdlineArgs.batch_size = cmdlineArgs.batch_size // world_size

    print("Global Rank", globalrank, "World size", world_size, "Batch size", cmdlineArgs.batch_size)

    if not cmdlineArgs.random_data:
        print("using cifar10 dataset")


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
            root=cmdlineArgs.data_root, train=True, download=True, transform=train_transform
        )


        test_dataset = CIFAR10(
            root=cmdlineArgs.data_root, train=False, download=True, transform=test_transform
        )
        nlabels = 10

    else:
        print("using randomly generated data")
        nlabels = cmdlineArgs.random_nlabels




    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=globalrank)

    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cmdlineArgs.batch_size,
                shuffle=False, # set shuffle to false because the DistributedSampler already shuffles by default
                sampler=train_sampler,
                drop_last=True,
                num_workers=cmdlineArgs.workers,
                pin_memory= True
            )
    test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cmdlineArgs.batch_size,
                shuffle=False,
                num_workers=cmdlineArgs.workers,
            )



    run_results = []
    for _ in range(cmdlineArgs.n_runs):

        print("Trying to make model on globalrank", globalrank)

        Model = getattr(torchvision.models, cmdlineArgs.model)
        model = DDP(Model(num_classes=nlabels).to(localrank), device_ids=[localrank],bucket_cap_mb=cmdlineArgs.bucket_cap)

        print(f"model successfully build on globalrank {globalrank}")

        optimizer = optim.SGD(model.parameters(), lr=cmdlineArgs.lr, momentum=0)

        if globalrank == 0:
            av_time_per_epoch = 0

        for epoch in range(1, cmdlineArgs.epochs + 1):

            train_sampler.set_epoch(epoch)
            if (globalrank == 0) and (epoch > 1):
                t_start_epoch = time.time()

            av_losses = train(model, localrank, train_loader, optimizer, epoch)

            if (globalrank == 0) and (epoch > 1):
                t_end_epoch = time.time()
                av_time_per_epoch += t_end_epoch - t_start_epoch

        if (globalrank==0) and (cmdlineArgs.epochs>1):
            if cmdlineArgs.write_scaling_results:
                nodename = os.environ.get('MASTER_ADDR')
                if cmdlineArgs.random_data:
                    dataset = "random"
                else:
                    dataset = "cifar"
                outputfile = open(cmdlineArgs.results_root+dataset+"-"+nodename+"-"+cmdlineArgs.scaling_type+"-"+str(world_size)+"-"+str(cmdlineArgs.batch_size)+".pyout","w+")
                outputfile.write(f"node name:{nodename}\n")
                outputfile.write(f"batch-size: {cmdlineArgs.batch_size}\nepochs: {cmdlineArgs.epochs}\nprocesses per node: {cmdlineArgs.processes_per_node}\n")
                outputfile.write(f"comm backend: {cmdlineArgs.comm_backend}\nscaling type: {cmdlineArgs.scaling_type}\nbucket-cap: {cmdlineArgs.bucket_cap}\n")
                outputfile.write(f"model = {cmdlineArgs.model}\n")
                outputfile.write(f"number of samples in train set: {len(train_dataset)}\n")
                outputfile.write(f"num GPUs: {world_size}\n")
                outputfile.write("\n\n\nresults\n\n\n")

                if cmdlineArgs.random_data:
                    outputfile.write(f"dimension of (square) input image: {cmdlineArgs.random_data_dim}\n")
                    outputfile.write(f"num labels of random data: {cmdlineArgs.random_nlabels}\n")

            av_time_per_epoch /= (cmdlineArgs.epochs-1)
            print(f"\n\n\nav_time_per_epoch={av_time_per_epoch}")
            print(f"number of samples in train set: {len(train_dataset)}")
            print(f"num GPUs: {world_size}")
            print(f"Av samples processed per second per GPU: {(len(train_dataset)/av_time_per_epoch)/world_size}")
            tendexample = time.time()
            print(f"total running time of example() on globalrank 0: {tendexample-tstartexample}")
            print("\n\n\n")
            if cmdlineArgs.write_scaling_results:
                outputfile.write(f"av_time_per_epoch={av_time_per_epoch}\n")
                outputfile.write(f"final av loss = {av_losses}\n")
                outputfile.write(f"Av samples processed per second per GPU: {(len(train_dataset)/av_time_per_epoch)/world_size}\n")
                outputfile.write(f"total running time of example() on globalrank 0: {tendexample-tstartexample}\n")
                outputfile.close()


        run_results.append(test(model, localrank, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"resnet_{cmdlineArgs.lr}_"
        f"{cmdlineArgs.batch_size}_{cmdlineArgs.epochs}"
    )


    if cmdlineArgs.saveModelResults:
        torch.save(run_results, f"run_results_{repro_str}.pt")

    if cmdlineArgs.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")



def main():
    t0 = time.time()

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
        default=1,
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
        "--saveModelResults",
        action="store_true",
        default=False,
        help="Save the model results (default: false)",
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

    parser.add_argument(
        "--processes_per_node",
        default=1,
        type=int,
        metavar="P",
        help="num processes to run on each node (1 for each GPU)",
    )
    parser.add_argument(
        "--comm_backend",
        default="gloo",
        type=str,
        help="communication backend for PyTorch distributed",
    )
    parser.add_argument(
        "--scaling-type",
        default="strong",
        type=str,
        help="scaling type to measure, weak or strong",
    )
    parser.add_argument(
        "--bucket_cap",
        default=25,
        type=int,
        help="bucket cap for DDP",
    )
    parser.add_argument(
        "--write_scaling_results",
        default=True,
        type=int,
        help="Whether to create an output file summarizing the results",
    )
    parser.add_argument(
        "--results_root",
        default="./",
        type=str,
        help="where you want the results file to be saved",
    )
    parser.add_argument(
        "--random_data",
        action="store_true",
        default=False,
        help="use random data rather than cifar10"
    )
    parser.add_argument(
        "--random_data_dim",
        default=32,
        type=int,
        help="image dims of random data, which will be square (default: 32)"
    )
    parser.add_argument(
        "--random_data_num",
        default=100000,
        type=int,
        help="number of data points for random data (defaults to 100k)"
    )
    parser.add_argument(
        "--random_nlabels",
        default=10,
        type=int,
        help="number of labels for random data  (default: 10)"
    )
    parser.add_argument("--model",
                        default="resnet18",
                        type=str,
                        help="specify a model from torchvision (default is resnet18)"
                        )

    cmdlineArgs = parser.parse_args()


    if cmdlineArgs.random_data:
        train_dataset = MyRandomDataSet(cmdlineArgs.random_data_dim,cmdlineArgs.random_data_dim,cmdlineArgs.random_data_num,cmdlineArgs.random_nlabels)
        test_dataset = MyRandomDataSet(cmdlineArgs.random_data_dim,cmdlineArgs.random_data_dim,1,cmdlineArgs.random_nlabels)
    else:
        train_dataset,test_dataset = None,None

    processes_per_node = cmdlineArgs.processes_per_node
    numNodes = int(os.environ.get('SLURM_NNODES'))
    nodeid = int(os.environ.get('SLURM_NODEID'))
    world_size = numNodes*processes_per_node

    print(f"spawning {processes_per_node} processes on node {nodeid}")

    if world_size == None:
        print("Error: missing world size")
    mp.spawn(example,
             args=(world_size,nodeid,cmdlineArgs, train_dataset, test_dataset),
             nprocs=processes_per_node,
             join=True)

    t1 = time.time()
    print(f"running time on node {nodeid}: {t1-t0}")



if __name__ == "__main__":
    main()
