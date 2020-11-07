#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size: int, word_embed_size: int, kernel_size: int = 5, padding: int = 1):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv1d = nn.Conv1d(in_channels=char_embed_size,
                                out_channels=word_embed_size,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=True)


    def forward(self, x: Tensor) -> Tensor:
        """
        x is tensor of shape (batch_size, char_embed_size, max_word_length)
        """
        x_conv = self.conv1d(x)  # (batch_size, word_embed_size, max_word_length - kernel_size + 1)
        x_conv_out, _ = torch.max(F.relu(x_conv), dim=-1)  # (batch_size, char_embed_size)
        return x_conv_out

    ### END YOUR CODE

def main():
    char_embed_size = 3
    word_embed_size = 50
    max_char_length = 2
    batch_size = 5
    cnn = CNN(char_embed_size, word_embed_size)
    x_reshaped = torch.randn(batch_size, char_embed_size, max_char_length + 2)
    x_conv_out = cnn(x_reshaped)
    assert x_conv_out.size() == (batch_size, word_embed_size)
    print("CNN shape check passed!")

if __name__ == '__main__':
    main()