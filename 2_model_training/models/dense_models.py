"""
Dense Neural Network Models for Cell Cycle Prediction
======================================================

This module contains dense (fully connected) neural network architectures:
- SimpleDenseModel (DNN3): 3-layer architecture, identified as top performer
- DeepDenseModel (DNN5): 5-layer deeper architecture

Author: Halima Akhter
Date: 2025-11-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDenseModel(nn.Module):
    """
    Simple 3-layer Dense Neural Network (DNN3)

    This is the top-performing model according to benchmark results.
    Architecture: input -> 128 -> 64 -> num_classes

    Parameters:
    -----------
    input_dim : int
        Number of input features (genes)
    num_classes : int
        Number of output classes (3 for G1, S, G2M)

    Attributes:
    -----------
    fc1 : nn.Linear
        First fully connected layer (input_dim -> 128)
    fc2 : nn.Linear
        Second fully connected layer (128 -> 64)
    fc3 : nn.Linear
        Output layer (64 -> num_classes)
    dropout1 : nn.Dropout
        Dropout layer with p=0.5 after first layer
    dropout2 : nn.Dropout
        Dropout layer with p=0.3 after second layer
    """

    def __init__(self, input_dim, num_classes):
        super(SimpleDenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

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
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class DeepDenseModel(nn.Module):
    """
    Deep 5-layer Dense Neural Network (DNN5)

    Deeper architecture with more parameters for complex pattern learning.
    Architecture: input -> 256 -> 128 -> 64 -> num_classes

    Parameters:
    -----------
    input_dim : int
        Number of input features (genes)
    num_classes : int
        Number of output classes (3 for G1, S, G2M)

    Attributes:
    -----------
    fc1 : nn.Linear
        First fully connected layer (input_dim -> 256)
    fc2 : nn.Linear
        Second fully connected layer (256 -> 128)
    fc3 : nn.Linear
        Third fully connected layer (128 -> 64)
    fc4 : nn.Linear
        Output layer (64 -> num_classes)
    dropout1 : nn.Dropout
        Dropout layer with p=0.5 after first layer
    dropout2 : nn.Dropout
        Dropout layer with p=0.4 after second layer
    dropout3 : nn.Dropout
        Dropout layer with p=0.3 after third layer
    softmax : nn.Softmax
        Softmax activation for output probabilities
    """

    def __init__(self, input_dim, num_classes):
        super(DeepDenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

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
            Output probabilities of shape (batch_size, num_classes) after softmax
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


class AttentionLayer(nn.Module):
    """Attention layer for enhanced dense model"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        out = x * attention_weights  # Element-wise multiplication with attention weights
        return out


class EnhancedDenseModel(nn.Module):
    """
    Enhanced Dense Model (from original code)

    Flexible architecture with variable number of layers.
    """
    def __init__(self, n_layers, units_per_layer, dropouts, input_features):
        super(EnhancedDenseModel, self).__init__()

        if len(units_per_layer) != n_layers or len(dropouts) != n_layers:
            raise ValueError("Mismatched lengths of units_per_layer or dropouts")

        layers = []
        in_features = input_features
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, units_per_layer[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
            in_features = units_per_layer[i]
        layers.append(nn.Linear(in_features, 3))  # 3 output classes

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EnhancedDenseAttentionModel(nn.Module):
    """
    Enhanced Dense Model with Attention (from original code)

    Adds attention layer to focus on important input features.
    """
    def __init__(self, n_layers, units_per_layer, dropouts, input_features):
        super(EnhancedDenseAttentionModel, self).__init__()

        if len(units_per_layer) != n_layers or len(dropouts) != n_layers:
            raise ValueError("Mismatched lengths of units_per_layer or dropouts")

        layers = []
        # Attention layer to focus on important input features
        layers.append(AttentionLayer(input_features))

        in_features = input_features
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, units_per_layer[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
            in_features = units_per_layer[i]

        # Final output layer for classification
        layers.append(nn.Linear(in_features, 3))  # 3 output classes

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
