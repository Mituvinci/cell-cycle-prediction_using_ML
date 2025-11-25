"""
Convolutional Neural Network Models for Cell Cycle Prediction
==============================================================

This module contains 1D CNN architectures for processing gene expression data.

Author: Halima Akhter
Date: 2025-11-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network for Cell Cycle Prediction

    Uses 1D convolutions to capture local patterns in gene expression profiles.
    Architecture: Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> output

    Parameters:
    -----------
    input_dim : int
        Number of input features (genes)
    num_classes : int
        Number of output classes (3 for G1, S, G2M)

    Attributes:
    -----------
    conv1 : nn.Conv1d
        First convolutional layer (1 -> 32 channels)
    conv2 : nn.Conv1d
        Second convolutional layer (32 -> 64 channels)
    pool : nn.MaxPool1d
        Max pooling layer with kernel size 2
    fc1 : nn.Linear
        First fully connected layer
    fc2 : nn.Linear
        Output layer
    dropout1 : nn.Dropout
        Dropout with p=0.1 after first conv layer
    dropout2 : nn.Dropout
        Dropout with p=0.3 after second conv layer
    dropout3 : nn.Dropout
        Dropout with p=0.5 for fully connected layer
    """

    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU()

        # Dropout layers with different rates
        self.dropout1 = nn.Dropout(p=0.1)  # Early features, retain more
        self.dropout2 = nn.Dropout(p=0.3)  # Intermediate features
        self.dropout3 = nn.Dropout(p=0.5)  # FC, regularize heavily

        # Compute final dimension after conv and pooling
        conv_output_dim = 64 * (input_dim // 4)
        self.fc1 = nn.Linear(conv_output_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the network

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)

        x = self.fc2(x)
        return x
