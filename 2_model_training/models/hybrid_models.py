"""
Hybrid Neural Network Models for Cell Cycle Prediction
=======================================================

This module contains hybrid architectures combining different techniques:
- HybridCNNDenseModel: CNN feature extraction + dense classification
- FeatureEmbeddingModel: Learned embedding + dense layers

Author: Halima Akhter
Date: 2025-11-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNDenseModel(nn.Module):
    """
    Hybrid CNN-Dense Model

    Combines convolutional feature extraction with dense classification layers.
    Architecture: Conv1D -> Pool -> Dense layers -> output

    Parameters:
    -----------
    input_dim : int
        Number of input features (genes)
    output_dim : int
        Number of output classes (3 for G1, S, G2M)
    conv_out_channels : int, default=64
        Number of output channels from CNN layer
    kernel_size : int, default=3
        Size of convolutional kernel
    dense_units : list of int, default=[128, 64]
        Number of units in each dense layer
    dropouts : list of float, default=[0.3, 0.3]
        Dropout rates for each dense layer

    Attributes:
    -----------
    conv1 : nn.Conv1d
        Convolutional layer for feature extraction
    pool : nn.MaxPool1d
        Max pooling layer
    dense_layers : nn.ModuleList
        List of dense layers with ReLU and dropout
    output_layer : nn.Linear
        Final classification layer
    """

    def __init__(self, input_dim, output_dim, conv_out_channels=64, kernel_size=3,
                 dense_units=[128, 64], dropouts=[0.3, 0.3]):
        super(HybridCNNDenseModel, self).__init__()

        # CNN layer
        self.conv1 = nn.Conv1d(1, conv_out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Dense layers after CNN
        self.dense_layers = nn.ModuleList()
        in_features = (input_dim // 2) * conv_out_channels  # Account for pooling
        for i in range(len(dense_units)):
            self.dense_layers.append(nn.Linear(in_features, dense_units[i]))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(dropouts[i]))
            in_features = dense_units[i]

        # Output layer
        self.output_layer = nn.Linear(in_features, output_dim)

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
            Output log probabilities of shape (batch_size, output_dim)
        """
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for dense layers

        for layer in self.dense_layers:
            x = layer(x)

        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class FeatureEmbeddingModel(nn.Module):
    """
    Feature Embedding Model

    Learns a lower-dimensional embedding of input features before classification.
    Architecture: Embedding -> Dense layers -> output

    Parameters:
    -----------
    input_dim : int
        Number of input features (genes)
    embed_dim : int
        Dimension of the learned embedding space
    n_layers : int
        Number of dense layers after embedding
    units_per_layer : list of int
        Number of units in each dense layer
    dropouts : list of float
        Dropout rates for each dense layer
    output_dim : int
        Number of output classes (3 for G1, S, G2M)

    Attributes:
    -----------
    embedding : nn.Linear
        Linear layer for feature embedding
    network : nn.Sequential
        Sequential network of dense layers with ReLU and dropout
    """

    def __init__(self, input_dim, embed_dim, n_layers, units_per_layer, dropouts, output_dim):
        super(FeatureEmbeddingModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Define dense layers after embedding
        layers = []
        in_features = embed_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, units_per_layer[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
            in_features = units_per_layer[i]

        # Final layer for classification
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

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
            Output logits of shape (batch_size, output_dim)
        """
        x = self.embedding(x)
        x = self.network(x)
        return x
