import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor """
    def __init__(self, in_size, out_size, h1=256, h2=256, h3=256):
        super(ActorNet, self).__init__()
        self.l1 = nn.Linear(in_size, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, h3)
        self.l4 = nn.Linear(h3, out_size)
        self.r = nn.ReLU()
        self.s = nn.Sigmoid()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.r(self.l2(x))
        x = self.r(self.l3(x))
        return self.s(self.l4(x))


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, h1=512, h2=512, h3=512):

        super(CriticNet, self).__init__()
        self.l1 = nn.Linear(state_size, h1)
        self.l2 = nn.Linear(h1 + action_size, h2)
        self.l3 = nn.Linear(h2, h3)
        self.l4 = nn.Linear(h3, 1)
        self.r = nn.ReLU()

    def forward(self, stuff):
        x, a = stuff
        x = self.r(self.l1(x))
        x = self.r(self.l2(torch.cat([x, a], 1)))
        x = self.r(self.l3(x))
        return self.l4(x)
