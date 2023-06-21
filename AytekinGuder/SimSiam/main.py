import argparse
import math
import os

import torch
import numpy as np
import random
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from simsiam.loader import CIFAR10, IMAGENET100
from simsiam.model_factory import SimSiam, Regressor
from simsiam.validation import KNNValidation
from simsiam.criterion import SimilarityLoss


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset', choices=["imagenet100", "cifar10"])
    parser.add_argument('--exp_dir', type=str, default='exp', help='path to experiment directory')

    parser.add_argument('--arch', default='resnet18', help='model name is used for training')

    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, help='learning rate for the encoder', dest='lr')
    parser.add_argument('--mlp-lr', '--mlp-learning-rate', default=0.0001, type=float, help='learning rate for the regressor', dest='mlp_lr')
    parser.add_argument('--lambda', default=1e-4, type=float, help='learning rate for the regressor', dest='lambda_')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    return parser.parse_args()


def main():
    set_seed(502)

    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.exp_dir, exist_ok=True)

    if args.dataset == "cifar10":
        train_set = CIFAR10(root=args.dataset,
                            train=True,
                            download=True
        )
    else:
        train_set = IMAGENET100(root=args.dataset,
                                train=True,
                                eval_=False
        )

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True
    )
    model = SimSiam(args).to(args.device)
    model_opt = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
    )

    regressor = Regressor(feature_dim=args.feat_dim).to(args.device)
    mlp_opt = optim.SGD(regressor.parameters(),
                        lr=args.mlp_lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
    )
    criterion = SimilarityLoss()

    #pbar = tqdm(range(args.epochs), dynamic_ncols=True)
    validation = KNNValidation(args, model.encoder)

    loss_hist, acc_hist = list(), list()

    for epoch in range(args.epochs):
        decay_learning_rate(model_opt, epoch, args.lr, args)
        decay_learning_rate(mlp_opt, epoch, args.mlp_lr, args)

        # train for one epoch
        train_loss = train(train_loader, model, regressor, criterion, model_opt, mlp_opt, epoch, args)
        loss_hist.append(train_loss.item())

        #pbar.set_description(f"Loss: {train_loss:.4f}")

        if (epoch+1) % 5 == 0:
            top1_acc = validation.eval()
            save_checkpoint(args, epoch, model, regressor, model_opt, mlp_opt, top1_acc)
            acc_hist.append(top1_acc)

            print(f"Epoch: [{epoch+1}/{args.epochs}] => Loss: {train_loss:.4f}, Acc: {top1_acc: .4f}")
        else:
            print(f"Epoch: [{epoch+1}/{args.epochs}] => Loss: {train_loss:.4f}")

    return loss_hist, acc_hist


def train(train_loader, model, regressor, criterion, model_opt, mlp_opt, epoch, args):
    # switch to train mode
    model.train()

    total_loss, total_num = 0.0, 0

    for g1, g2, l1, l2 in tqdm(train_loader, leave=False, dynamic_ncols=True):
        N = g1.shape[0]

        g1, g2 = g1.to(args.device), g2.to(args.device)
        l1, l2 = l1.to(args.device), l2.to(args.device)

        zl1, zl2 = model.encoder(l1), model.encoder(l2)
        zg1, zg2 = model.encoder(g1), model.encoder(g2)

        k = torch.randperm(n=N) # hope no j=k

        # optimize regressor
        regressor.train()
        omega = regressor.omega_loss(zl1, zl2, zl1[k])

        omega.backward()
        mlp_opt.step()
        mlp_opt.zero_grad()

        l_gg = criterion(model.predictor(zg1), zg2)

        pl1, pl2 = model.predictor(zl1), model.predictor(zl2)

        l_lg = criterion(pl1, zg1)
        l_lg += criterion(pl1, zg2)
        l_lg += criterion(pl2, zg1)
        l_lg += criterion(pl2, zg2)

        regressor.eval()
        l_ll = regressor(zl1, zl2).mean()

        loss = l_gg + l_lg + args.lambda_ * l_ll

        loss.backward()
        model_opt.step()
        model_opt.zero_grad()

        total_num += N
        total_loss += loss * N

    return total_loss / total_num


def save_checkpoint(args, epoch, model, regressor, model_opt, mlp_opt, acc):
    state = {
        'epoch': epoch,
        'arch': args.arch,

        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),

        'regressor': regressor.state_dict(),
        'regressor_opt': mlp_opt.state_dict(),

        'top1_acc': acc
    }
    torch.save(state, os.path.join(args.exp_dir, f"ckpt_epoch_{epoch}.pt"),)


# lr scheduler for training
def decay_learning_rate(optimizer, epoch, lr, args):
    """
        lr = min_lr + 0.5*(max_lr - min_lr) * (1 + cos(pi * t/T))
    """
    lr = 0.5 * lr * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    loss, acc = main()
    print(loss)
    print(acc)