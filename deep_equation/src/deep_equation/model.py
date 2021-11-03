import math
import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, operation_embedding_size=64):
        super(Model, self).__init__()
        self.classes = classes()
        self.digit_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.operation_stack = nn.Sequential(
            nn.Linear((2*64*5*5) + operation_embedding_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(),
            nn.Linear(512, len(self.classes)))
        self.operation_embedding = nn.Embedding(4, operation_embedding_size)

    def forward(self, digits_a, digits_b, operations):
        digits_a = self.digit_stack(digits_a)
        digits_b = self.digit_stack(digits_b)

        N, _, _, _ = digits_a.size()
        digits_a = digits_a.view(N, -1)
        digits_b = digits_b.view(N, -1)

        embedded_operations = self.operation_embedding(operations).reshape(N, -1)

        concat = torch.cat((digits_a, digits_b, embedded_operations), 1)
        output = self.operation_stack(concat)

        return output

def classes():
    A = np.array(range(10))
    B = np.array(range(10))

    add = np.array([format_result(a+b) for a in A for b in B])
    sub = np.array([format_result(a-b) for a in A for b in B])
    mul = np.array([format_result(a*b) for a in A for b in B])
    div = np.array([format_result(a/b) if b != 0 else math.nan for a in A for b in B])

    return np.unique(np.r_[add, sub, mul, div])

def format_result(value):
    return f'{float(value):>.5f}'
