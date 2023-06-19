# Author: Utku Mert Topçuoğlu (modified functions from OTA papers repo.)
import pickle
from icecream import ic
import time
import torch
from torch import nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.ops import roi_align
import random
from naive_box_generation import add_n_random_boxes

DEBUG = True
FILTER_SIZE = 64
K_COMMON_INSTANCES = 4

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

def xywh_to_xyxy(proposal_boxes):
    """
    Convert bounding boxes from (top-left x, top-left y, width, height) format
    to (x1, y1, x2, y2) format.
    Arguments:
    - proposal_boxes: A torch tensor of shape (K, 4), where K is the number of boxes,
                      and each box is represented by [x, y, w, h].
    Returns:
    - A torch tensor of shape (K, 4), where each box is represented by [x1, y1, x2, y2].
    """
    # Extract x, y, width and height
    x, y, w, h = proposal_boxes[:, 0], proposal_boxes[:, 1], proposal_boxes[:, 2], proposal_boxes[:, 3]
    # Compute x2 and y2
    x2 = x + w
    y2 = y + h
    # Stack the coordinates to get the final tensor
    return torch.stack([x, y, x2, y2], dim=1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        return x if random.random() > self.p else self.fn(x)

def crop_scene(image, image_size):
    """
    NOTE With mode='bicubic', it’s possible to cause overshoot, in other words it can produce negative values or values greater than 1 for images (0-1 ranging images).
     Explicitly call result.clamp(min=0, max=1) if you want to reduce the overshoot when displaying the image.
    """
    # The git author missed different augmentations: https://github.com/lucidrains/byol-pytorch/issues/31#issuecomment-707925576
    # Apply random resized crop
    rand_res_crop = T.RandomResizedCrop((image_size, image_size))
    top, left, height, width = rand_res_crop.get_params(image, rand_res_crop.scale, rand_res_crop.ratio)
    # if size is int (not list) smaller edge will be scaled to match this.
    # byol uses bicubic interpolation.
    image = TF.resized_crop(image, top, left, height, width, size=(image_size,  image_size), interpolation=TF.InterpolationMode.BICUBIC).clamp(min=0, max=1)
    return image, (top, left, height, width)

def common_augmentations(image, type_two = False):
    """Could not use Compose with transformations due to RandomApply requirement."""
    image = T.RandomHorizontalFlip(p=0.5)(image)
    # since the order has to change (with randperm i get_params) I must use ColorJitter below.
    image = (RandomApply(T.ColorJitter(0.4, 0.4, 0.2, 0.1), p = 0.8)(image)).clamp(min=0, max=1) # NOTE color jitter produces NaN values if the input image is not between 0-1.
    # Apply grayscale with probability 0.2
    image = (T.RandomGrayscale(p=0.2)(image)).clamp(min=0, max=1)
    # Apply gaussian blur with probability 0.2
    gaussian_prob = 0.1 if type_two else 1 # 1 for type_one
    image = (RandomApply(T.GaussianBlur((23, 23)), p = gaussian_prob)(image)).clamp(min=0, max=1)
    solarize_prob = 0.2 if type_two else 0 # assymetric augm
    solarize_threshold = 0.5
    image = (T.RandomSolarize(threshold=solarize_threshold, p=solarize_prob)(image)).clamp(min=0, max=1)
    # NOTE toTensor not necessary because images are read with torchvision.io.read()
    # They apply normalization (not explicit in the paper: https://github.com/lucidrains/byol-pytorch/issues/4#issue-641183816)
    image = (T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))(image)).clamp(min=0, max=1)
    return image


# 1. Get scene overlaps given the coordinates
def get_scene_overlap(crop_coordinates_one, crop_coordinates_two):
    min_overlap_size = FILTER_SIZE # filter scenes with too small overlap , NOTE might cause difference with original paper
    """ crop_coordinates are in t,l,h,w form (https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)
    return scene as (x1, y1, x2, y2)"""
    def tlhw_to_xyxy_single(t,l,h,w):
        x1 = l
        y1 = t
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2
    def get_overlap(coord_one, coord_two):
        x1_1, y1_1, x2_1, y2_1 = coord_one
        x1_2, y1_2, x2_2, y2_2 = coord_two
        x1 = max(x1_1, x1_2) # left
        y1 = max(y1_1, y1_2) # top
        x2 = min(x2_1, x2_2) # right
        y2 = min(y2_1, y2_2) # bottom
        return (x1, y1, x2, y2)

    coord_one = tlhw_to_xyxy_single(*crop_coordinates_one)
    coord_two = tlhw_to_xyxy_single(*crop_coordinates_two)
    (x1, y1, x2, y2) = get_overlap(coord_one, coord_two)
    # NOTE ideally you want: return None if (x2 - x1 < min_overlap_size or y2 - y1 < min_overlap_size) else (x1, y1, x2, y2)
    return None if (x2 - x1 < min_overlap_size or y2 - y1 < min_overlap_size) else (x1, y1, x2, y2)


def check_box_in_region(overlap_region, proposal_boxes):
    """Check if boxes are inside the region fully.
    overlap_region has (x,y,w,h), proposal_boxes has (x1,y1,x2,y2) content"""
    xyxy_bs = xywh_to_xyxy(proposal_boxes) 
    r_x1, r_y1, r_x2, r_y2 = overlap_region
    return (xyxy_bs[:, 0] >= r_x1) & (xyxy_bs[:, 1] >= r_y1) & (xyxy_bs[:, 2] <= r_x2) & (xyxy_bs[:, 3] <= r_y2)


def get_overlapping_boxes(overlap_region, proposal_boxes):
    """ Get the proposed boxes in the overlapping region."""
    inside_region_mask = check_box_in_region(overlap_region=overlap_region, proposal_boxes=proposal_boxes)
    overlapping_boxes = proposal_boxes[inside_region_mask]
    return overlapping_boxes if len(overlapping_boxes)!=0 else torch.zeros(0, 4, dtype=torch.int64)


# 2. If they have at least K_common_instances object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)
def select_scenes(img, proposal_boxes, image_size, K_common_instances=K_COMMON_INSTANCES, iters=20):
    # NOTE we get only K_common_instances boxes and ablations show there is no improvement after 4!!
    """Returns scenes with at least K_common_instances common targets in the overlapping regions. Has to be applied to each image individually (not batch), blocking operations are not parallelizable effectively."""
    best_scenes={"overlapping_boxes":[], "overlap_coord":None, "s1":None, "s2":None}
    while iters > 0:
        # I need the information which regions of the images were cropped and if RandomHorizontalFlip was applied (the region will change accordingly.)
        scene_one, crop_coordinates_one  = crop_scene(img, image_size)
        scene_two, crop_coordinates_two = crop_scene(img, image_size)
        overlap_coord = get_scene_overlap(crop_coordinates_one, crop_coordinates_two)
        if overlap_coord is None: # Check there is a large enough overlap
            continue
        # now check K_common_instances common instances.
        overlapping_boxes = get_overlapping_boxes(overlap_region=overlap_coord, proposal_boxes=proposal_boxes)
        if len(overlapping_boxes) >= K_common_instances:
            return scene_one, scene_two, overlapping_boxes[:K_common_instances] # Get only first K_common_instances boxes.
        elif len(overlapping_boxes) >= len(best_scenes["overlapping_boxes"]): # Update the best boxes for the fallback case.
            best_scenes["overlapping_boxes"], best_scenes["overlap_coord"], best_scenes["s1"], best_scenes["s2"] = overlapping_boxes, overlap_coord, scene_one, scene_two
        iters -= 1
    else:
        # Add random boxes to the overlapping coordinates.
        missing_box_num = K_common_instances-len(best_scenes["overlapping_boxes"])
        best_scenes["overlapping_boxes"] = add_n_random_boxes(overlap_coord=best_scenes["overlap_coord"], overlapping_boxes=best_scenes["overlapping_boxes"], n_random_boxes=missing_box_num)
        return best_scenes["s1"], best_scenes["s2"], best_scenes["overlapping_boxes"][:K_common_instances] # Get only first K_common_instances boxes.


def get_concatenated_instances(img, overlapping_boxes):
    if img.ndim==3:
        img, overlapping_boxes = img.unsqueeze(dim=0), overlapping_boxes.unsqueeze(dim=0)
    # Resize and feed instances in overlapping boxes to the online encoder
    # Interpolation was not specified, hence can use bilinear interpolation as in roi_align.
    # img shape should be (b, c, h, w) - batch, channels, height, width
    # overlapping_boxes shape should be (b, n, 4) - batch, number of boxes, coordinates (x1, y1, x2, y2)

    # Number of boxes per image
    num_boxes = overlapping_boxes.size(1)
    
    # Create batch indices to be concatenated with boxes -> (batch_size*K), each box will have an index (showing where it belongs)
    batch_indices = torch.arange(img.size(0)).view(-1, 1).repeat(1, num_boxes).view(-1, 1).to(img)
    
    # Reshape boxes for roi_align
    boxes = overlapping_boxes.view(-1, 4) # Collect total number of boxes in the first dim (batch_size*K)
    # NOTE for roi_align to work, overlapping boxes has to be converted to xyxy format.
    boxes = xywh_to_xyxy(boxes).to(img)
    
    # Concatenate batch indices with boxes, index shows which image in a batch each box belongs
    boxes_with_indices = torch.cat([batch_indices, boxes], dim=1).to(img)
    
    # Crop and resize using roi_align
    output_size = (96, 96)
    instances = roi_align(img, boxes_with_indices, output_size)
    return instances # Now instances tensor has shape (n, c, 96, 96)