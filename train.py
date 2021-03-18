# For Training Steps

import torch
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()

def loss(loss_func,logits,target):
    return loss_func(logits,target)


def train():
    pass

def trainIters():
    # for training iterations
    pass


