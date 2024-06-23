import torch
import torch.nn as nn
import torch.nn.functional as F


def cnn_model(printtoggle=False, output_layers=26):
    class emnistnet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()
            self.print = printtoggle

            # * FEATURE MAP LAYERS
            # First convolution layer with 1 input channel, 6 output channels, and a 3x3 kernel with padding of 1
            self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(6)  # Number of channels in this layer

            # Second convolution layer with 6 input channels, 6 output channels, and a 3x3 kernel with padding of 1
            self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
            self.bnorm2 = nn.BatchNorm2d(6)  # Number of channels in this layer

            # * LINEAR DECISION LAYER
            # Fully connected layer with 294 input features and 50 output features
            self.fc1 = nn.Linear(7 * 7 * 6, 50)
            # Fully connected layer with 50 input features and 26 output feature results
            self.fc2 = nn.Linear(50, output_layers)

        def forward(self, x):
            if self.print:
                print(f"Input: {list(x.shape)}")

            # First block: convolution -> maxpool -> batchnorm -> relu
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.leaky_relu(self.bnorm1(x))
            if self.print:
                print(f"First CPR block: {list(x.shape)}")

            # Second block: convolution -> maxpool -> batchnorm -> relu
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.leaky_relu(self.bnorm2(x))
            if self.print:
                print(f"Second CPR block: {list(x.shape)}")

            # Reshape for linear layer
            nUnits = x.shape.numel()/x.shape[0]
            x = x.view(-1, int(nUnits))
            if self.print:
                print(f"Vectorized: {list(x.shape)}")

            # Linear layers
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)
            if self.print:
                print(f"Final output: {list(x.shape)}")

            return x

    # Create model instance
    net = emnistnet(printtoggle)
    # Loss function
    lossfun = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)

    return net, lossfun, optimizer
