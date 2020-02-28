import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
device = torch.device('cpu')

class AMAGOLD(nn.Module):
    def __init__(self, trainloader, mh_trainloader, datasize, T, weight_decay, beta, criterion):
        super(AMAGOLD, self).__init__()
        self.datasize = datasize
        self.T = T 
        self.weight_decay = weight_decay
        self.beta = beta
        self.criterion = criterion
        self.trainloader = trainloader
        self.mh_trainloader = mh_trainloader

    def outer_loop(self,model,p_buf,lr):
        sig = 0
        model_old = deepcopy(model)
        model, U_new, U_old, rho, p_buf, neg_p_buf_old = self.leapfrog(model,model_old,p_buf,lr)
        a = torch.exp((U_old - U_new)*self.datasize + rho)
        u = torch.rand(1)
        if u.item()<=a.data.item():
            sig = 1
            return sig,model,p_buf
        else:
            return sig,model_old,neg_p_buf_old

    def update_params(self,model,t,p_buf,lr):
        rho = 0.0
        i = 0
        neg_p_buf_old = []
        for p in model.parameters():
            if t == 0:
                p.data += 0.5 * p_buf[i]
            else:                
                d_p = p.grad.data
                d_p.add_(self.weight_decay, p.data)
                eps = torch.randn(p.size()).to(device)
                buf_old = deepcopy(p_buf[i])
                neg_p_buf_old.append(-buf_old)
                p_buf[i] = ((1 - self.beta) * p_buf[i] - lr * d_p + (lr*self.beta)**.5*2*eps)/(1+self.beta)
                rho += torch.sum(d_p * (buf_old + p_buf[i])) 
                if t == self.T-1:
                    p.data += 0.5*p_buf[i]
                else:
                    p.data += p_buf[i]
            i += 1
        return 0.5*rho, p_buf, neg_p_buf_old

    def leapfrog(self,model,model_old,p_buf,lr):
        model.train()
        t = 0
        rho = 0
        for _ in range(self.T):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(device), target.to(device)
                if t>0:
                    model.zero_grad()
                    output = model(data)
                    loss = self.criterion(output, target)*self.datasize
                    loss.backward()
                if t == 1:
                    rho0, p_buf, neg_p_buf_old = self.update_params(model,t,p_buf,lr)
                else:
                    rho0, p_buf, _ = self.update_params(model,t,p_buf,lr)
                rho += rho0
                t += 1
                if t == self.T:
                    break
            if t == self.T:
                break

        for batch_idx, (data, target) in enumerate(self.mh_trainloader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            U_new = self.criterion(output, target)
            model_old.zero_grad()
            output = model_old(data)
            U_old = self.criterion(output, target)
            break

        return model, U_new, U_old, rho, p_buf, neg_p_buf_old