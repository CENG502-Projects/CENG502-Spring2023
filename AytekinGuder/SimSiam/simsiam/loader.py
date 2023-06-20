# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from PIL import Image, ImageFilter
import random

class CIFAR10(CIFAR10):
    def __init__(self, root, train, download, r_g=0.65, r_l=0.4):
        super().__init__(root, train, transform=None, download=download)

        self.global_transform = self.augmentation((r_g, 0.9))
        self.local_transform = self.augmentation((0.15, r_l))

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    def augmentation(self, scale):
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        g1 = self.global_transform(img)
        g2 = self.global_transform(img)
        l1 = self.local_transform(img)
        l2 = self.local_transform(img)

        return g1, g2, l1, l2

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class IMAGENET100(datasets.ImageFolder):
    def __init__(self, root, train=True, eval_=False, r_g=0.65, r_l=0.4):
        super().__init__(root=f"{root}/{'train' if train else 'val'}", transform=None)

        self.eval = eval_
        self.base_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.global_transform = self.augmentation((r_g, 0.9))
        self.local_transform = self.augmentation((0.15, r_l))

    def augmentation(self, scale):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if not self.eval: # SSL mode, no targets
            g1 = self.global_transform(img)
            g2 = self.global_transform(img)
            l1 = self.local_transform(img)
            l2 = self.local_transform(img)

            return g1, g2, l1, l2

        else:
            img = self.base_transforms(img)
            return img, target

if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader

    d = IMAGENET100("imagenet100")
    # loader = DataLoader(d, batch_size=8, shuffle=True)

    # for g1, g2, l1, l2 in loader:
    #     print(g1.shape, g2.shape, l1.shape, l2.shape)
    #     break