# Author: Utku Mert Topçuoğlu (took help from chatgpt https://chat.openai.com/share/caa323fa-babc-4b26-9faf-028cf10f740b.)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.io as io
import pickle
from icecream import ic
import time
from transforms import select_scenes, get_concatenated_instances, K_COMMON_INSTANCES
import matplotlib.pyplot as plt
import random

DEBUG = True
ORI_FILTERED_PKL = "/raid/utku/datasets/COCO_dataset/COCO_proposals/final_proposals/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou.pkl"
TRIAL_FILTERED_PKL = '/raid/utku/datasets/COCO_dataset/COCO_proposals/trial/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou_trial_250.pkl'
DATASET_PATH = "/raid/utku/datasets/COCO_dataset/train2017/"
IMAGE_SIZE = 224
SMALL_IMGS = ['000000187714.jpg', '000000363747.jpg']

if not DEBUG:
    ic.disable()

previous_timestamp = int(time.time())
def unixTimestamp():
    global previous_timestamp
    current_timestamp = int(time.time())
    minutes, seconds = divmod(current_timestamp-previous_timestamp, 60)
    previous_timestamp = current_timestamp
    return 'Local time elapsed %02d:%02d |> ' % (minutes, seconds)

ic.configureOutput(prefix=unixTimestamp)

# Check if the boxes correspond to correct images (sorted)
def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data


class CustomDataset(Dataset):
    def __init__(self, filtered_proposals:dict, image_size=IMAGE_SIZE):
        filtered_proposals_bbox = filtered_proposals["bbox"]
        for small_img in SMALL_IMGS: # Remove small images (below 64)
            del filtered_proposals_bbox[small_img]
        self.filtered_proposals_bbox = list(filtered_proposals_bbox.items())
        self.total_sample_count=len(self.filtered_proposals_bbox)
        self.image_size=image_size
        scene_one, _, concatenated_instances = self.__getitem__(0)
        assert scene_one.dtype == torch.float32 
        assert concatenated_instances.shape[0]==K_COMMON_INSTANCES
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return self.total_sample_count # or the number of image paths

    def __getitem__(self, idx):
        # Load your data here using the idx
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id # NOTE To reduce load to different gpus and increase num of workers
        img_name, proposal_boxes = self.filtered_proposals_bbox[idx] 
        img_path = DATASET_PATH + img_name
        # Load the image using torchvision's read_image, reads int8 (cv2 also reads so)
        img = io.read_image(img_path)/255 # convert to 0-1 float32
        # NOTE feeding non-normalized float image values to ColorJitter clamps all values to max(x,1.0)!!!
        
        # NOTE Would be faster in GPU probably, worth to try later.
        scene_one, scene_two, overlapping_boxes = select_scenes(img=img,proposal_boxes=proposal_boxes,image_size=self.image_size) # return scene_one, scene_two, overlapping_boxes
        scene_one, scene_two, overlapping_boxes = scene_one.to(worker_id), scene_two.to(worker_id), overlapping_boxes.to(worker_id)
        concatenated_instances = get_concatenated_instances(img, overlapping_boxes)
        if scene_one.shape[0] == 1:
            scene_one, scene_two, concatenated_instances = scene_one.expand(3, -1, -1), scene_two.expand(3, -1, -1), concatenated_instances.expand(K_COMMON_INSTANCES, 3, -1, -1)
        
        # scene_one, scene_two, overlapping_boxes = scene_one.to("cpu"), scene_two.to("cpu"), overlapping_boxes.to("cpu")
        return (scene_one, scene_two, concatenated_instances)
    
    def random_sample(self):
        # Randomly select an index
        random_idx = random.randint(0, self.total_sample_count - 1)
        # Get scene_one, scene_two, concatenated_instances for the randomly selected index
        return self.__getitem__(random_idx)


def init_dataset(batch_size, ddp=False):
    # Load the filtered box proposal pkl files, convert to faster readable form (tensors?)
    ic("load proposals to memory")
    FILTERED_PROPOSALS = load_pkl(pkl_file=ORI_FILTERED_PKL) # TODO chenge if necessary, TRIAL_FILTERED_PKL or ORI_FILTERED_PKL

    # Initialize your dataset
    dataset = CustomDataset(FILTERED_PROPOSALS)
    num_samples = len(dataset)
    sampler = None
    if ddp:
        # Initialize DistributedSampler
        # https://discuss.pytorch.org/t/distributedsampler/90205/2?u=utku_mert_topcuoglu
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True) # Basically random sampler for distributed training.
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8) # sampler or shuffle.
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, sampler, num_samples

def visualize_scenes(scene_one, scene_two, concatenated_instances, save_path):
    """
    Function to visualize scene_one, scene_two and concatenated_instances.
    scene_one, scene_two: Tensors of shape (C, H, W)
    concatenated_instances: Tensor of shape (K, C, H, W)
    """

    # Convert tensors to numpy arrays for visualization
    scene_one_np = scene_one.permute(1, 2, 0).cpu().numpy()
    scene_two_np = scene_two.permute(1, 2, 0).cpu().numpy()

    # Plot the scenes
    plt.figure(figsize=(16, 6))

    plot_img(1, scene_one_np)
    plt.title('Scene One')

    plot_img(4, scene_two_np)
    plt.title('Scene Two')

    # Plotting concatenated_instances
    for count,plt_idx in enumerate([2,3,5,6]):  # Loop through concatenated instances (up to 4)
        instance = concatenated_instances[count].permute(1, 2, 0).cpu().numpy()
        plot_img(plt_idx, instance)
        plt.title(f'Concatenated Instance {plt_idx+1}')

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_img(idx, img_np):
    plt.subplot(2, 3, idx)
    plt.imshow(img_np)
    plt.axis('off')

def vis_some_samples(samle_num, dataloader, log_dir):
    for i in range(samle_num):
        scene_one, scene_two, concatenated_instances = dataloader.dataset.random_sample()
        vis_path = log_dir/"vis";vis_path.mkdir(exist_ok=True)
        visualize_scenes(scene_one, scene_two, concatenated_instances, save_path=str(vis_path/f"ss_i_pair_{i}.jpg"))