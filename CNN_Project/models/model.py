import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class MyNeuralNet(LightningModule):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(MyNeuralNet, self).__init__()  # Initialize the parent class

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_features, 32, 3)  # 28x28x1 -> 26x26x32
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 26x26x32 -> 13x13x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13x13x32 -> 11x11x64
        self.relu2 = nn.LeakyReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(11 * 11 * 64, 256)  # Adjust the input size based on your data
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, out_features)
        self.softmax = nn.Softmax(dim=1)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        # import pdb
        # pdb.set_trace()

        # Flatten the output before passing through fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00045)
        return optimizer
