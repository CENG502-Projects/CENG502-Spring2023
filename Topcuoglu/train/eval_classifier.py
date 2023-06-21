""" They use random resized crop for training 
https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/benchmarks/linear_classification/imagenet/r50_multihead_28ep.py#L24"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import models
from typing import OrderedDict
import os
from icecream import ic
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader_IN import ImageNetKaggleDataset

DEVICE = 0 # main device rank.
EPOCHS = 28
NUM_CLASSES = 1000
NUM_WORKERS = 8
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = 256
LOCAL_BATCH_SIZE = 256 // WORLD_SIZE
IMAGE_SIZE = 224
UVIP_PRETRAIN_PT_PATH = "/home/kuartis-dgx1/utku/UniVIP/train/uni_vip_pretrained_model.pt"
EVAL_ONLY = False # No train only evaluation.
LR_SCALAR = (LOCAL_BATCH_SIZE*WORLD_SIZE)/GLOBAL_BATCH_SIZE # normally should be 1.
LR = 0.01*LR_SCALAR

LOG_DIR = Path(f"/home/kuartis-dgx1/utku/UniVIP/classification_logs_{WORLD_SIZE}")
LOG_DIR.mkdir(exist_ok=True)
LAST_EPOCH_FILE = LOG_DIR/"last_epoch.txt"
CHECKPOINT_PATH = LOG_DIR/"uvip_imagenet_linear_prob.pt"

def ddp_setup(rank, world_size):
    # initialize the process group,  It ensures that every process will be able to coordinate through a master, using the same ip address and port. 
    # nccl backend is currently the fastest and highly recommended backend when using GPUs. This applies to both single-node and multi-node distributed training. https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html 
    # NOTE might set ,timeout= to avoid any timeout during random box selection (might lose synchronization)
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size) # For multi GPU train.

def cleanup():
    dist.destroy_process_group()

# Helper functions
def log_text(file, content):
    # Log the last epoch
    with open(file, "w") as f:
        f.write(str(content))


def save_model(rank, model, cur_epoch, avg_epoch_loss, writer, accuracy=None):
    # save your improved network
    if rank == DEVICE:
        ic("saving network")
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), str(CHECKPOINT_PATH))
        # Log the last epoch
        log_text(str(LAST_EPOCH_FILE),cur_epoch)
        if avg_epoch_loss:
            writer.add_scalar("Avg Loss", avg_epoch_loss, cur_epoch)

# Define the ResNet model
def get_single_resnet():
    # If you're in a distributed environment, make sure all processes are synchronized before loading
    # state_dict = model.state_dict()
    state_dict = torch.load(UVIP_PRETRAIN_PT_PATH, map_location=lambda storage, loc: storage.cuda(DEVICE))
    new_state_dict = {}

    module_key_prefix = "module.online_encoder.net."
    for name, param in state_dict.items():
        if name.startswith(module_key_prefix):
            new_state_dict[name.replace(module_key_prefix, "")] = param

    new_linear_layer = torch.nn.Linear(2048, 1000)
    new_state_dict["fc.weight"] = new_linear_layer.weight
    new_state_dict["fc.bias"] = new_linear_layer.bias

    new_state_dict = OrderedDict(new_state_dict)
    resnet = models.resnet50(pretrained=False)
    # Assign the updated state_dict to the model
    resnet.load_state_dict(new_state_dict)

    # Freeze previous layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Enable training for the new linear layer
    for param in resnet.fc.parameters():
        param.requires_grad = True
    resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet) # Not sure if UniVIP does this.
    return resnet


def get_ddp_resnet(single_resnet, rank):
    single_resnet = single_resnet.to(rank)
    ddp_resnet = DDP(single_resnet, device_ids=[rank])
    ddp_resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_resnet) # Not sure if UniVIP does this.
    return ddp_resnet


def get_transforms():
    # Copied from ORL (says UniVIP) https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/benchmarks/linear_classification/imagenet/r50_multihead_28ep.py#L24
    # They added this strange lightning noise which I neglected: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/openselfsup/datasets/pipelines/transforms.py#L44
    train_transforms =  transforms.Compose([
                        transforms.RandomResizedCrop(size=IMAGE_SIZE),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transforms =    transforms.Compose([
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=IMAGE_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transforms, val_transforms

def get_dataloaders(rank):
    if rank == DEVICE:
        ic("load data")
    # Define the dataset and data loader
    train_transforms, val_transforms = get_transforms()
    train_dataset = ImageNetKaggleDataset(split="train",transform=train_transforms)
    val_dataset = ImageNetKaggleDataset(split="val",transform=val_transforms)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=LOCAL_BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=LOCAL_BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS)
    return train_loader, train_sampler, val_loader

def train_eval(rank, world_size, single_resnet, only_eval=False):
    # Define the model, loss function, and optimizer
    ddp_setup(rank=rank, world_size=world_size)
    ddp_resnet = get_ddp_resnet(single_resnet=single_resnet, rank=rank)
    # Set up the distributed environment
    torch.cuda.set_device(rank)
    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_loader, train_sampler, val_loader = get_dataloaders(rank)
    if not only_eval:
        train(ddp_resnet, rank, train_sampler, train_loader)
    eval(ddp_resnet, rank, val_loader)

    
def train(ddp_resnet, rank, train_sampler, train_loader):
    # Dataloaders and train sampler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    # test saving.
    if rank == DEVICE:
        writer = SummaryWriter(log_dir=str(LOG_DIR))
        save_model(rank=rank, model=ddp_resnet, cur_epoch=0, avg_epoch_loss=0, writer=writer)
    # Training loop
    num_epochs = 28
    if rank == DEVICE:
        ic("Start training")
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        ddp_resnet.train()
        total_epoch_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True) if rank==DEVICE else train_loader
        for images, labels in progress_bar_train:
            images = images.to(rank)
            labels = labels.to(rank)
            # Forward pass
            outputs = ddp_resnet(images)
            loss = criterion(outputs, labels)
            total_epoch_loss += loss.detach()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Learning rate scheduling
        lr_scheduler.step()
        if rank == DEVICE:
            avg_epoch_loss = total_epoch_loss/len(train_loader) # returns ceil(length / self.batch_size)
            save_model(rank=rank, model=ddp_resnet, cur_epoch=epoch, avg_epoch_loss=avg_epoch_loss, writer=writer)
        # Print training progress
        if rank == 0:
            ic(f"Epoch [{epoch+1}/{num_epochs}] completed.")
    ic("Finished training!")

def eval(ddp_resnet, rank, val_loader):
    # Evaluation on validation split
    ddp_resnet.eval()
    total_samples = 0
    correct_predictions = 0
    if rank == DEVICE:
        ic("Start validation")
    progress_bar_val = tqdm(val_loader, desc="Validate", position=0, leave=True) if rank==DEVICE else val_loader
    for images, labels in progress_bar_val:
        images = images.to(rank)
        labels = labels.to(rank)
        
        with torch.no_grad():
            outputs = ddp_resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # NOTE calculate accuracy across all ranks (globally in DDP):
    torch.distributed.barrier()
    # Wrap total_samples and correct_predictions as tensors
    correct_predictions_tensor = torch.tensor(correct_predictions).to(rank)
    total_samples_tensor = torch.tensor(total_samples).to(rank)

    torch.distributed.barrier()
    # Sum the tensors across all ranks
    dist.all_reduce(correct_predictions_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    torch.distributed.barrier()
    # Calculate accuracy on rank 0
    if rank == DEVICE:
        # Convert tensors back to python scalars
        global_correct_predictions = correct_predictions_tensor.item()
        global_total_samples = total_samples_tensor.item()
        global_accuracy = global_correct_predictions / global_total_samples * 100
        results_txt = f"Global Top-1 Accuracy: {global_accuracy:.4f}%"
        ic(results_txt)
        with open(f"eval_results_{WORLD_SIZE}.txt", "w") as eval_writer:
            eval_writer.write(results_txt)
    cleanup()

def main():
    # Set the number of GPUs and spawn multiple processes
    single_resnet = get_single_resnet()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357' # set port for communication
    mp.spawn(train_eval, nprocs=WORLD_SIZE, args=(WORLD_SIZE,single_resnet,EVAL_ONLY), join=True)
    
if __name__ == '__main__':
    main()
