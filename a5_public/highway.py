#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embed_size: int):
        super(Highway, self).__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate = nn.Linear(word_embed_size, word_embed_size, bias=True)

    """
    x is input of shape (batch_size, word_embed_size)
    """

    def forward(self, x: Tensor) -> Tensor:
        x_proj = F.relu(self.proj(x))  # (batch_size, word_embed_size)
        x_gate = torch.sigmoid(self.gate(x))  # (batch_size, word_embed_size)
        x_highway = x_gate * x_proj + (1 - x_gate) * x  # (batch_size, word_embed_size)
        return x_highway

    ### END YOUR CODE


def main():
    word_embed_size = 3
    batch_size = 5
    highway = Highway(word_embed_size)
    x_proj_out = torch.rand(batch_size, word_embed_size)
    assert highway(x_proj_out).size(), (5, 3)
    print("Highway module tensor shape check passed!")


if __name__ == '__main__':
    main()
