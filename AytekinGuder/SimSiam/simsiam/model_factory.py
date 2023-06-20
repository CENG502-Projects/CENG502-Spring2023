import torch
from torch import nn
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torchvision.models

class LoGoBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Regressor(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64):
        super().__init__()
        blocks = [
            LoGoBlock(in_dim=2*feature_dim, out_dim=hidden_dim),
            LoGoBlock(in_dim=hidden_dim, out_dim=hidden_dim),
            LoGoBlock(in_dim=hidden_dim, out_dim=hidden_dim),
            LoGoBlock(in_dim=hidden_dim, out_dim=hidden_dim),
            LoGoBlock(in_dim=hidden_dim, out_dim=hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, z1, z2):
        return self.net(torch.cat([z1, z2], axis=-1))

    def omega_loss(self, zl1, zl2, zl_neg):
        zl1 = zl1.detach()
        zl2 = zl2.detach()
        zl_neg = zl_neg.detach()

        return -(self.forward(zl1, zl2).mean() - self.forward(zl1, zl_neg).mean())


class projectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()

        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class predictionMLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()

        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, args):
        super(SimSiam, self).__init__()

        self.backbone = SimSiam.get_backbone(args.arch, args.dataset)

        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = projectionMLP(out_dim, args.feat_dim, args.num_proj_layers)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.predictor = predictionMLP(args.feat_dim)

    @staticmethod
    def get_backbone(backbone_name, dataset):
        if dataset=="cifar10":
            return {'resnet18': ResNet18(),
                    'resnet34': ResNet34(),
                    'resnet50': ResNet50(),
                    'resnet101': ResNet101(),
                    'resnet152': ResNet152()}[backbone_name]
        else:
            return {'resnet18': torchvision.models.resnet18(),
                    'resnet34': torchvision.models.resnet34(),
                    'resnet50': torchvision.models.resnet50(),
                    'resnet101': torchvision.models.resnet101(),
                    'resnet152': torchvision.models.resnet152()}[backbone_name]