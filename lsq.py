# -*- coding: utf-8 -*-
# learned step size quantization
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter

import numpy as np

from collections import OrderedDict

# v- = clip(v/s. -Q_N, Q_p).round()
# v^ = v-*s
# s => a

# +
## method 2
# class LSQ(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, weight, alpha, g, Qn, Qp): # quantization하는 과정
#         assert alpha > 0, 'alpha = {}'.format(alpha)
#         ctx.save_for_backward(weight, alpha)
#         ctx.other = g, Qn, Qp
#         q_w = (weight / alpha).round().clamp(Qn, Qp)
#         w_q = q_w * alpha
#         return w_q

#     @staticmethod
#     def backward(ctx, grad_weight):
#         weight, alpha = ctx.saved_tensors
#         g, Qn, Qp = ctx.other
#         q_w = weight / alpha
#         indicate_small = (q_w < Qn).float()
#         indicate_big = (q_w > Qp).float()
#         # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
#         indicate_middle = 1.0 - indicate_small - indicate_big # Thanks to @haolibai 
#         grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
#                 -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
#         grad_weight = indicate_middle * grad_weight
#         return grad_weight, grad_alpha, None, None, None

# +
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad # forward : y , backward : scale


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad # forward : round(x) - x , backward : 1


# -

class Q_ReLU(nn.Module) : # n; quantization bit, q_n ; 0 , q_v ; positive => non_negative distribution
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.a = nn.Parameter(torch.Tensor(1)) # a - trainable parameter - step size parameter
        self.q_lv = 0
        
        self.act_func = act_func
        self.inplace = inplace
        
        self.Qn = 0
        self.Qp = 0
        
    def initialize(self, q_lv, input): 
        self.q_lv = q_lv
        self.Qn = 0
        self.Qp = q_lv-1 # 2**a_bit -1
        self.a.data.copy_(2 * input.abs().mean() / math.sqrt(self.Qp))
    '''
    The authors use Tensor(v.abs().mean() * 2 / sqrt(Qp)) as initial values of the step sizes 
    in both weight and activation quantization layers, where Qp is the upper bound of the quantization space, 
    and v is the initial weight values or the first batch of activations.
    '''
    def forward(self, x):
        if self.act_func:
            x = F.relu(x, self.inplace) # 우선 relu로 한번 거름
        
        if self.q_lv == 0:
            return x
        else:
            g = 1.0 / math.sqrt(x.numel() * self.Qp)
            a = grad_scale(self.a,g)
            x = round_pass((x/a).clamp(self.Qn,self.Qp))*a
#             method 2
#             x = LSQ.apply(x, self.a, g, self.Qn, self.Qp) # def forward(ctx, weight, alpha, g, Qn, Qp)
            return x


class Q_Linear(nn.Linear): 
    def __init__(self, *args, **kwargs): # *args = (weghts, bias)
        super(Q_Linear, self).__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.Tensor(1))
        self.q_lv = 0 
        self.Qn = 0
        self.Qp = 0
        self.q_weight = 0
        
    def initialize(self, q_lv):
        self.q_lv= q_lv
        self.Qn = (-1) * (q_lv//2 -1) # symmetric 하게 유지하고 싶어서 q_p = -q_n 으로 사용
        self.Qp = q_lv//2 - 1
        self.a.data.copy_(2 * self.weight.abs().mean() / math.sqrt(self.Qp))
        
        
    def _weight_quant(self, alpha, Qn, Qp):        
#         method 2
#         g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)
#         weight = LSQ.apply(self.weight, self.a, g, self.Qn, self.Qp) # def forward(ctx, weight, alpha, g, Qn, Qp)
        weight = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        
        return weight
    
    def forward(self, x):
        if self.q_lv == 0:
            return F.linear(x, self.weight, self.bias) # qunatization 안하는 부분
        else:
            g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)
            a = grad_scale(self.a, g)
            
            weight = self._weight_quant(a, self.Qn, self.Qp) # quantize 된 weight
            self.q_weight = weight
            return F.linear(x, weight, self.bias) # qunatization 하는 부분


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Q_Conv2d, self).__init__(*args, **kwargs)
        self.q_lv = 0
        self.a = nn.Parameter(torch.Tensor(1))
        self.weight_old = None
        self.Qn = 0
        self.Qp = 0
        self.q_weight = 0
        
    def initialize(self, q_lv):
        self.q_lv = q_lv
        self.Qn = (-1) * (q_lv//2 -1) # symmetric 하게 유지하고 싶어서 q_p = -q_n 으로 사용
        self.Qp = q_lv//2 - 1
        self.a.data.copy_(2 * self.weight.abs().mean() / math.sqrt(self.Qp))
        
        
    def _weight_quant(self, alpha, Qn, Qp):
#         method 2
#         g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)
#         weight = LSQ.apply(self.weight, self.a, g, self.Qn, self.Qp) # def forward(ctx, weight, alpha, g, Qn, Qp)
        weight = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        
        return weight
    
    def forward(self, x):
        if self.q_lv == 0:
            return F.conv2d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)
        else:
            g = 1.0 / math.sqrt(self.weight.numel() * self.Qp) # grad scaler
            a = grad_scale(self.a, g) # forward : a*g, backward : g
            
            weight = self._weight_quant(a, self.Qn, self.Qp)

            return F.conv2d(x, weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)

# +
import math

# act = activation quantization, weight = weight qunatization
def initialize(model, loader, q_lv, act=False, weight=False, printed=False): 
    Qn = (-1) * (q_lv//2 -1)
    Qp = q_lv//2 - 1
    def initialize_hook(module, input, output): # layer module 마다 적용 -> qunatization q_lv 초기화
        if isinstance(module, Q_ReLU) and act: # activation만 quantization
            if not isinstance(input, torch.Tensor): 
                input = input[0]
            
            module.initialize(q_lv, input) # activation parameter 's' initialize

        
        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight: # CONV2와 Linear에 들어오기 전에 quantizaiton 실행
            module.initialize(q_lv)
            if printed : quant_check(module, q_lv, Qn, Qp) # quantization check
            
    hooks = []
    
    for name, module in model.named_modules(): 
        hook = module.register_forward_hook(initialize_hook) 
        hooks.append(hook) 
        
    model.train() # train_mode
    model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break
        
    model.cuda()
    for hook in hooks:
        hook.remove() 


# -

class Q_Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Q_Sequential, self).__init__()
        
        if len(args) == 1 and isinstance(args[0], Orderdict): # sequential이 orderdict로 선언된 경우
            for key, module in args[0].items():
                self.add_module(key, module) # name : key, module = module
        else:
            idx = 0
            for module in args:
                if isinstance(module, Q_Sym) or (isinstance(module, Q_HSwish) and idx==0):
                    self.add_module('-'+str(idx), module)
                else:
                    self.add_module(str(idx),module) # module 그냥 저장
                    idx += 1


class QuantOps(object):
    initialize = initialize
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU   
    Sequential = Q_Sequential
    Linear = Q_Linear


def make_hist(p):
    n = p.cpu().detach().numpy()
    # print(n.shape)
    hist = {}
    size = 1
    for i in n.shape:
        size *= i

    l = n.reshape(size)
    for i in l:
        if str(i) in hist:
            hist[str(i)] += 1
        else : hist[str(i)] = 1

    keys = []
    for key in hist:
        keys.append(float(key))
    keys.sort()
#     print(keys)
    for i in range(len(keys)-1):
        sub = keys[-1]-keys[-2]
        if keys[-1-i]-keys[-2-i] - sub > 1e-5:
            print("이상")

    return hist


def quant_check(module, q_lv, Qn, Qp):
#     print(module)
    p = module._weight_quant(module.a, Qn, Qp)
    hist = make_hist(p)
    if len(hist) == q_lv-1 : print("quant OK")
    else : 
        print('q_lv = ', q_lv)
        print('len(hist) = ', len(hist))
        print(hist)
        print('warning')
        sum = 0
        for keys in hist :
            sum += hist[keys]
        print("sum = ", sum)
