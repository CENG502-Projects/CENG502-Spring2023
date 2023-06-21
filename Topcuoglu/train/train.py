# Author: Utku Mert Topçuoğlu (modified from https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py#LL267C1-L268C27.)
"""
Run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train_script.py

NOTE to load DDP weights: configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
"""
import os
from pathlib import Path
from uni_vip import UVIP
import torch
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from UniVIP.train.dataloader_uvip import init_dataset, vis_some_samples
from math import ceil

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from icecream import ic

LOG_DIR = Path("/home/kuartis-dgx1/utku/UniVIP/logs")
LOG_DIR.mkdir(exist_ok=True)
LAST_EPOCH_FILE = LOG_DIR/"last_epoch.txt"
CHECKPOINT_PATH = LOG_DIR/"uni_vip_pretrained_model.pt"
LOAD_CHECKPOINT = True
VISUALIZE_SAMPLE_NUM = 20
VISUALIZE = False
DEBUG = True

if DEBUG: # TODO last changes slew down the code abnormally.
    torch.autograd.set_detect_anomaly(True) # TODO re-run with this to detect problem with optimizer steps.

# DDP train settings.
USE_DDP = True
DEVICE = 0 # Device for single gpu training
WORLD_SIZE = 8 # Number of GPUs for multi gpu training

# was not pretrained by default for ORL also.https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L7 
batch_size = 64 # 512 for COCO training
total_epochs = 800 # 800 for COCO and COCO+ training
# update momentum every iteration with cosine annealing.
base_learning_rate = 0.2 # same as ORL.
final_min_lr = 0 # Here it was said 0, no explicit in univip: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L144



# Helper functions
def log_text(file, content):
    # Log the last epoch
    with open(file, "w") as f:
        f.write(str(content))

def update_lr(optimizer, lr):
    # Update the learning rate of the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(rank, model, cur_epoch, avg_epoch_loss, writer):
    # save your improved network
    if rank == DEVICE:
        ic("saving network")
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), str(CHECKPOINT_PATH))
        # Log the last epoch
        log_text(str(LAST_EPOCH_FILE),cur_epoch)
        writer.add_scalar("Avg Loss", avg_epoch_loss, cur_epoch)

def load_ddp_model(model, rank):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
    # There is a mismatch with the pretrained weight variables. Because projector is created afterwards.
    model.load_state_dict({k: state_dict[k] for k in state_dict if k in model.state_dict()})


def ddp_setup(rank, world_size):
    # initialize the process group,  It ensures that every process will be able to coordinate through a master, using the same ip address and port. 
    # nccl backend is currently the fastest and highly recommended backend when using GPUs. This applies to both single-node and multi-node distributed training. https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html 
    # NOTE might set ,timeout= to avoid any timeout during random box selection (might lose synchronization)
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size) # For multi GPU train.

def cleanup():
    dist.destroy_process_group()


def train_single_epoch(model, progress_bar, optimizer, total_iterations, global_current_iteration, warm_up_iters, warm_up_lrs, rank):
    epoch_loss = 0
    # train for one epoch.
    for img_data in progress_bar:
        # Warmup first epochs.
        if global_current_iteration < warm_up_iters:
            lr = warm_up_lrs[global_current_iteration]
            update_lr(optimizer, lr=lr)
        global_current_iteration+=1
        scene_one, scene_two, concatenated_instances=(item.to(rank) for item in img_data)
        loss = model((scene_one, scene_two, concatenated_instances))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not torch.stack([torch.isnan(p).any() for p in model.parameters()]).any(), "model has nan" # TODO retrain for this
        epoch_loss += loss.item()
        # After each step teacher is updated based on BYOL paper.
        # NOTE can update_moving_average for every device, because each device has its own/same copy, hence no need for extra rank block
        if USE_DDP:
            model.module.update_moving_average(tot_iter=total_iterations,cur_iter=global_current_iteration)
        else:
            model.update_moving_average(tot_iter=total_iterations,cur_iter=global_current_iteration)
    # Log the loss
    return model, epoch_loss, global_current_iteration
    

def train_setup(rank, world_size):
    if USE_DDP:
        ddp_setup(rank=rank, world_size=world_size)
    # Create DataLoader with the custom dataset and the distributed sampler
    dataloader, sampler, num_samples = init_dataset(batch_size=batch_size, ddp=USE_DDP)
    iters_per_epoch = ceil(num_samples / batch_size)
    total_iterations = iters_per_epoch*total_epochs 
    global_current_iteration=0
    # warmup
    warm_up_epochs = 4
    warm_up_iters = iters_per_epoch*warm_up_epochs
    warm_up_lrs = [iter*base_learning_rate/warm_up_iters for iter in range(warm_up_iters)] # increase linearly

    # Create the model.
    resnet = models.resnet50(weights=None).to(rank)
    model = UVIP(resnet).to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # Not sure if UniVIP does this.
    # NOTE To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
    if USE_DDP:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) # will return ddp model output.
        # configure map_location properly
        if LOAD_CHECKPOINT:
            load_ddp_model(model=model, rank=rank)
    # save your network test
    if rank == DEVICE:
        writer = SummaryWriter(log_dir=str(LOG_DIR))
        # save_model(rank,model=model,cur_epoch=0, avg_epoch_loss=0, writer=writer)
    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0001, momentum=0.9, lr=base_learning_rate)
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs-warm_up_epochs, eta_min=final_min_lr)

    # Training epochs.
    for cur_epoch in range(total_epochs):
        # Shuffle data at the beginning of each epoch
        if USE_DDP:
            sampler.set_epoch(epoch=cur_epoch) # to ensure that the distributed sampler shuffles the data differently at the beginning of each epoch
        progress_bar = tqdm(dataloader, desc=f"Epoch {cur_epoch+1}/{total_epochs}", position=0, leave=True) if rank==DEVICE else dataloader
        model, epoch_loss, global_current_iteration = train_single_epoch(model, progress_bar, optimizer, total_iterations, global_current_iteration, warm_up_iters, warm_up_lrs, rank)
        # Cosine scheduler after some steps.
        # NOTE Do not apply cosine for the first warm_up_epochs, otherwise collides with warmup. First step considers base_lr not the current lr of the optimizer. Look here: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR:~:text=elif%20self._step_count,optimizer.param_groups)%5D
        if cur_epoch >= warm_up_epochs:
            cosine_scheduler.step(epoch=cur_epoch-warm_up_epochs) # epoch gives warning
        if rank == DEVICE:# save your improved network
            # NOTE I dont have to block here because backward pass synchronizes each rank/gpu process
            save_model(rank, model=model, cur_epoch=cur_epoch, avg_epoch_loss=epoch_loss/iters_per_epoch, writer=writer)
    if USE_DDP:
        cleanup() # For multi GPU train.


if __name__ == "__main__":
    # Randomly sample from the dataset
    # Visualize the scenes
    if VISUALIZE:
        temp_dataloader, sampler, num_samples  = init_dataset(batch_size=batch_size, ddp=False)
        vis_some_samples(samle_num=VISUALIZE_SAMPLE_NUM, dataloader=temp_dataloader, log_dir=LOG_DIR)
        del temp_dataloader.dataset  # Delete the dataset
        del temp_dataloader, sampler
    if USE_DDP:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # set port for communication
        mp.spawn(train_setup,args=(WORLD_SIZE,),nprocs=WORLD_SIZE, join=True) # NOTE rank is passed automatically to each call
    else:
        train_setup(rank=DEVICE, world_size=1) # single gpu train

