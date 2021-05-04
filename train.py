# -*- coding: utf-8 -*-
# +
## LSQ ##
# -

import os
import datetime
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
from models.resnet import *
from models.preactresnet import * 
from lsq import QuantOps
import pickle
import matplotlib.pyplot as plt


# +
# parser = argparse.ArgumentParser()
# parser.add_argument("--ckpt", help="checkpoint directory")
# parser.add_argument('--data', choices=['cifar10', 'cifar100'])
# parser.add_argument('--model', choices=['resnet18_10','resnet18_100',
#                     'preactresnet18_10','preactresnet18_100'])
# parser.add_argument('--gpu', choices=['0','1','2','3'])
# parser.add_argument('--quant_op', choices=['duq'])
# parser.add_argument("--lr", default=0.04, type=float)
# parser.add_argument("--decay", default=2e-5, type=float)
# parser.add_argument("--warmup", default=3, type=int)
# parser.add_argument("--ft_epoch", default=15, type=int)

# +
# args = parser.parse_args()

# +
class a :
    def __init__(self, ckpt, data, model, gpu, duq, a_bit, w_bit, lr, decay, warmup, ft_epoch):
        self.ckpt = ckpt
        self.data = data 
        self.model = model
        self.gpu = gpu
        self.quant_op = duq
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.lr = lr
        self.decay = decay
        self.warmup = warmup
        self.ft_epoch = ft_epoch

#################  
bit = 2
lr = 0.01
decay = 5e-4
warmup = 1

ft_epoch = 2
#################
args = a('./ckpt/','cifar100', 'resnet18_100', '0', 'lsq', [bit], [bit], lr, decay, warmup, ft_epoch)
# print(args.ckpt, args.data ,args.model ,args.gpu, args.quant_op, 
#         args.a_bit,args.w_bit, args.lr, args.decay)
ckpt_root = args.ckpt



# +
#data
if args.data == 'cifar10':
    root = './data/CIFAR10'
elif args.data == 'cifar100':
    root = './data/CIFAR100'
    
print("=====> dataset.{}".format(args.data))
# -

from cifardata import get_loader
trainloader, testloader = get_loader(root,train_batch=256,test_batch=100,num_workers=8)
print("====> get dataloader")

# +
# device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("====> device : {}".format(device))

# quanatization option
print('====> Quantization_option = {}'.format(args.quant_op))
if args.quant_op == 'duq':
    from duq import QuantOps
    print("====> differentiable and unified quantization method is selected..")
elif args.quant_op == 'lsq':
    from lsq import QuantOps
else:
    raise NotImplementedError

model = ResNet18_100(QuantOps)
checkpoint = torch.load("./ckpt/ts_resnet18_100_lsq_8_8_best.pth", map_location = 'cuda:0') 
model.load_state_dict(checkpoint,False)

model.cuda(device)
## gpu setting ok

criterion = nn.CrossEntropyLoss()


# -

def categorize_param(model, skip_list=()):
    quant = []
    bnbias = []
    weight = []
    
    for name, param in model.named_parameters():
            
        if name.endswith('.a') :# a
            quant.append(param)            
        elif len(param.shape) == 1 or name.endswith(".bias"): # bn의 bias와 일반 bias
            bnbias.append(param)
        else: # 일반 convolution weigth
            weight.append(param)    
        
    return (quant, weight, bnbias)    


def get_optimizer(params, train_quant, train_weight, train_bnbias):
    (quant, weight, bnbias) = params
    optimizer = optim.SGD([
        {'params': quant, 'weight_decay': 0., 'lr': args.lr * 100000 if train_quant else 0},
        # quant_param -> 학습이 안되는 것 같습니다.
        # param값이 fine tuning 하면서 어느정도 변하긴 하나 lr*1000000 주는 것 대비
        # 변화량이 너무 미미해서, 이게 training이 되는건지 안되는지 판단이 서질 않습니다....
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr  if train_bnbias else 0}, 
        {'params': weight, 'weight_decay': args.decay, 'lr': args.lr * 100 if train_weight else 0}, # 학습이 잘 됩니다.
    ], momentum=0, nesterov=False)
    return optimizer


def phase_prefix(a_bit, w_bit): # ckpt name
    prefix_base = "ts_%s_%s_" % (args.model, args.quant_op)
    return prefix_base + ("%d_%d" % (a_bit, w_bit))


import os
import re
import glob
import time
import torch
import shutil
import tempfile
import collections
import numpy as np 
import pathlib
import math
_print_freq = 50
_temp_dir = tempfile.mkdtemp()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0) # view
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]


def train_ts(train_loader, model, criterion, optimizer, epoch, metric_map={}):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
   
        loss_class = criterion(output, target_var)

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        losses.update(loss_class.data.item(), input.size(0)) # losses class에 loss update
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
           
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_class.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg #1 epoch당 losses의 avg만 return


def test(val_loader, model, criterion, epoch, train=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.train(train) # model.eval와 같음

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():        
            if not isinstance(model, torch.nn.DataParallel):
                input = input.cuda()
            target = target.cuda(non_blocking=True)        
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)      

        if isinstance(output, tuple):
            loss = criterion(output[0], target_var)
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # record loss and accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg, losses.avg # top1 accuracy, loss의 avg 리턴


def get_loss_acc(root, prefix='train'):
    files = glob.glob(os.path.join(root, "{}_*.ckpt".format(prefix))) # 전체파일리스트, "ts_model_quantOps_abit_wbit"
    num_arr = []
    for file in files: # 전체 파일 리스트중 해당되는 것만.
        num = re.search("{}_(\d+).ckpt".format(prefix), file)
        if num is not None:
            num = num.group(1)
            num_arr.append(int(num))

    num_arr.sort()
    loss_arr = []
    acc_arr = []
    for num in num_arr:
        checkpoint = torch.load(
            os.path.join(root,"{}_{}.ckpt".format(prefix, num)))
        loss = checkpoint["loss"]
        acc = checkpoint["acc"]
        best_acc = checkpoint["best_acc"]
        
        loss_arr.append(loss)
        acc_arr.append(acc)
    return loss_arr, acc_arr, best_acc


def train_epochs(optimizer, warmup_len, max_epochs, prefix, best_acc):

    last_epoch = -1
    best_acc = 0
    scheduler = CosineWithWarmup(optimizer, 
                        warmup_len=warmup_len, warmup_start_multiplier=0.1,
                        max_epochs=max_epochs, eta_min=1e-3)

    loss_arr = []
    acc_arr = []

    if last_epoch+1 == max_epochs : # ckpt가 모두 존재할때만 실행
        loss, acc, best_acc = get_loss_acc(args.ckpt, prefix) # loss, acc만 가져오기
        loss_arr.extend(loss)
        acc_arr.extend(acc)

    for epoch in range(last_epoch+1,max_epochs):
        ls = train_ts(trainloader, model, criterion, optimizer, epoch) # losses.avg 반환

        loss_arr.append(ls) #batch train으로 뽑아낸 loss 저장
    
        acc_base, _ = test(testloader, model, criterion, epoch) # test accuracy, test_loss
        acc_arr.append(acc_base)
        
        is_best = False
        
        if acc_base > best_acc:
            is_best = True
        
        best_acc = max(best_acc, acc_base)
        scheduler.step()
    return best_acc, loss_arr, acc_arr


# +
### main code ###

QuantOps.initialize(model, trainloader, 2**args.a_bit[0], act=True) # activation initialize

params = categorize_param(model) # fine tuning 전 param check
print(params[0]) # quant parameter
print(params[1]) # weigt

t_loss_arr = []
acc_arr = []
best_acc = 0

a_bit, w_bit = 32, 32

## activation quant

for a_bit in args.a_bit:
    prefix = phase_prefix(a_bit, w_bit) # prefix로 파일명 설정 ;"ts_model_quantOps_abit_wbit"
    print("==> Activation quantization, bit %d" % a_bit)
    for name, module in model.named_modules():
        if isinstance(module, (QuantOps.ReLU)):
            module.q_lv = 2 ** a_bit

    print("==> Fine-tuning")
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 

    best_acc, loss, acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix, best_acc)
    
    t_loss_arr.extend(loss)
    acc_arr.extend(acc)
# -

# fine tuning 후 paramter check
print(params[0])
print(params[1])

# +
with torch.no_grad():
    QuantOps.initialize(model, trainloader, 2**args.w_bit[0], weight=True) # weight quantization
    
for w_bit in args.w_bit:
    prefix = phase_prefix(a_bit, w_bit)
    print("==> Weight quantization, bit %d" % w_bit)
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantOps.Conv2d, QuantOps.Linear)):
            module.q_lv = 2 ** w_bit
    print("==> Fine-tuning")
    params = categorize_param(model)
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
    best_acc, loss, acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix, best_acc)
    
    t_loss_arr.extend(loss)
    acc_arr.extend(acc)
# -

print("==> Finish training.. {}'s best accuracy is {}".format(args.model, best_acc))


# +
def graph_plot(loss, acc, path, w_bit, a_bit):
    
    fig, ax = plt.subplots(2,1, figsize=(15, 7))

    end_ep = len(w_bit)*2*args.ft_epoch
    if len(w_bit)==1 : 
        arange = np.arange(0,end_ep,2)
    else : 
        arange = np.arange(0,end_ep,2)

    ax[0].plot(loss, label = 'loss', color='C0')
    ax[0].set_title("total loss")
    ax[0].set_ylabel("loss")
    ax[0].set_xticks(arange)
    ax[0].legend(loc='upper left')

    ax[1].plot(acc, label = 'accuracy', color='C1')
    ax[1].set_title("accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_xticks(arange)
    ax[1].legend(loc='upper left')

    plt.show
#     plt.savefig(path+' loss and accuarcy a_bit '+str(a_bit)+' w_bit '+str(w_bit)+ '.png')

path = './loss/'+args.model+'_'+args.quant_op+'_ft_'+str(args.ft_epoch)

# direct quantization
if directed:
    path = './loss/directed_Q_'+args.model+args.quant_op+str(args.ft_epoch)

print(args.quant_op, args.model, args.data, args.a_bit, args.ft_epoch)    
graph_plot(t_loss_arr,acc_arr, path, args.w_bit, args.a_bit)

# -








