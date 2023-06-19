import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, latent=512, activation='relu'):
        super().__init__()
        self.affine_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(latent),
        )

        self.value_head = nn.Linear(latent, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.affine_layers(x)


        value = self.value_head(x)
        return value
