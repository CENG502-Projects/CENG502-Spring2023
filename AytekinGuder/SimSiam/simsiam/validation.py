# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
from .loader import IMAGENET100

class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        if args.dataset == "cifar10":
            base_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = datasets.CIFAR10(root=args.dataset,
                                             train=True,
                                             download=False,
                                             transform=base_transforms)
            val_dataset = datasets.CIFAR10(root=args.dataset,
                                           train=False,
                                           download=False,
                                           transform=base_transforms)

            self.n_data = train_dataset.data.shape[0]

        else:
            train_dataset = IMAGENET100(args.dataset,
                                        train=True,
                                        eval_=True)
            val_dataset = IMAGENET100(args.dataset,
                                      train=False,
                                      eval_=True)

            self.n_data = len(train_dataset.samples)

        self.targets = train_dataset.targets
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True)
    @torch.no_grad()
    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, self.n_data], device=self.device)

        for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)

            # forward
            features = self.model(inputs)
            features = nn.functional.normalize(features)
            train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

        train_labels = torch.LongTensor(self.targets).to(self.device)

        total, correct = 0, 0

        for inputs, targets in self.val_dataloader:
            targets = targets.cuda(non_blocking=True)
            batch_size = inputs.size(0)
            features = self.model(inputs.to(self.device))

            dist = torch.mm(features, train_features)
            yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

            total += batch_size
            correct += retrieval.eq(targets.data).sum().item()

        return correct / total

    def eval(self):
        return self._topk_retrieval()
