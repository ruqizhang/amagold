from __future__ import print_function
import argparse
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import sys
import pickle
from amagold import AMAGOLD
device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x),dim=1)

def test(model, it):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(it,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=2500, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=5e-6, metavar='M',
                        help='beta')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print('lr',args.lr,'beta',args.beta)
    datasize = 60000
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    mh_trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=datasize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    weight_decay = 5e-4
    T = 10
    lr = args.lr/datasize
    criterion = nn.NLLLoss()
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/sgd_init_epoch3.pt'))
    succ = 0
    amagold_optimizer = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion)
    p_buf = []
    for p in model.parameters():
        p_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))

    for it in range(1, args.iters+1):
        sig,model,p_buf = amagold_optimizer.outer_loop(model,p_buf,lr)
        succ += sig
        if it%10==0:
            test(model, it)
            print(1.0*succ/it)
