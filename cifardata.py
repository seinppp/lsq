# +
# ready for cifar10, cifar100

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os

# -

def get_loader(root, train_batch=256, test_batch=100, num_workers=8):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if root == './data/CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)
    elif root == './data/CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader
