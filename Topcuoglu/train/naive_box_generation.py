# Author: Utku Mert Topçuoğlu (modified functions from OTA papers repo.)
"""Methods in this repo work only per image (not for batch of images)"""
import torch
from torch import Tensor
import numpy as np

def get_max_iou_tensorized(filtered_boxes: Tensor, candidate_boxes: Tensor) -> Tensor:
    """
    filtered_boxes : multiple coordinate for predict bounding boxes (x, y, w, h)
    candidate_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the max iou score about filtered_boxes and candidate_box
    """
    # Making sure candidate_boxes is 2D
    if len(candidate_boxes.shape) == 1:
        candidate_boxes = candidate_boxes.unsqueeze(0)

    # 1.get the coordinate of inters
    ixmin = torch.max(filtered_boxes[:, 0].unsqueeze(1), candidate_boxes[:, 0].unsqueeze(0))
    ixmax = torch.min(filtered_boxes[:, 0].unsqueeze(1) + filtered_boxes[:, 2].unsqueeze(1), 
                     candidate_boxes[:, 0].unsqueeze(0) + candidate_boxes[:, 2].unsqueeze(0))
    iymin = torch.max(filtered_boxes[:, 1].unsqueeze(1), candidate_boxes[:, 1].unsqueeze(0))
    iymax = torch.min(filtered_boxes[:, 1].unsqueeze(1) + filtered_boxes[:, 3].unsqueeze(1), 
                     candidate_boxes[:, 1].unsqueeze(0) + candidate_boxes[:, 3].unsqueeze(0))

    iw = torch.clamp(ixmax - ixmin, min=0.)
    ih = torch.clamp(iymax - iymin, min=0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (filtered_boxes[:, 2].unsqueeze(1) * filtered_boxes[:, 3].unsqueeze(1) + 
           candidate_boxes[:, 2].unsqueeze(0) * candidate_boxes[:, 3].unsqueeze(0) - inters)

    # 4. calculate the overlaps and find the max overlap between filtered_boxes and candidate_box
    iou = inters / uni
    return None if iou.numel() == 0 else torch.max(iou, dim=0).values

def generate_random_box(overlap_coord, overlapping_boxes, min_size=64, max_ratio=3, iou_threshold=0.5, max_trials=100):
    """overlap_coord is (x1,y1,x2,y2), overlapping_boxes is (x,y,w,h)
    NOTE Removed imags with smaller edges than 64.
    NOTE max_trials might introduce worse results ->  after some iterations neglect the iou rule to avoid being stuck.
    max_ratio has to be >1 version of the ratio.
    Generate random x,y,w,h uniformly in the allowed ranges.
    iou_thresh, max_ratio and min_size as given.
    """
    device = overlapping_boxes.device
    overlapping_boxes= overlapping_boxes.to("cpu")
    (o_x1,o_y1,o_x2,o_y2)=overlap_coord
    max_width, max_height = o_x2-o_x1, o_y2-o_y1
    min_size = min(min(min_size, max_width), max_height) # The max values might be smaller than min_size.

    # Generate max_trials random boxes
    # generate width uniformly between min_size and max_width
    widths = np.random.randint(min_size, max_width + 1, max_trials)
    # generate heights  uniformly between max(min_size,width/max_ratio) and min(max_height,widht*max_ratio)
    min_heights = np.maximum(min_size, widths / max_ratio)
    max_heights = np.minimum(max_height, widths * max_ratio)
    min_heights = np.minimum(min_heights, max_heights) # has to be smaller than max_heights

    heights = np.random.randint(min_heights, max_heights + 1, max_trials)
    # generate random x1 and y1 uniformly (top-left coordinates) such that o_x1<=x1 and x1+width<=o_x2 and o_y1<=y1 and y1+height<o_y
    x1s = np.random.randint(o_x1, (o_x2 - widths + 1), max_trials)
    y1s = np.random.randint(o_y1, (o_y2 - heights + 1), max_trials)

    # Random boxes
    random_boxes = torch.tensor(np.stack([x1s, y1s, widths, heights], axis=1), dtype=overlapping_boxes.dtype)
    overlapping_boxes, random_boxes = overlapping_boxes.to(device), random_boxes.to(device)

    # Calculate IoUs for all boxes
    ious = get_max_iou_tensorized(overlapping_boxes, random_boxes)
    if ious is None:
        random_box = random_boxes[0]
    else:
        # Select a box with an IoU less than iou_threshold if exists, otherwise select the one with the least IoU
        acceptable_indices = torch.where(ious < iou_threshold)[0]
        if acceptable_indices.numel() > 0:
            random_box = random_boxes[acceptable_indices[0]]
        else:
            least_iou_index = torch.argmin(ious)
            random_box = random_boxes[least_iou_index]
    return random_box

def add_n_random_boxes(overlap_coord,overlapping_boxes,n_random_boxes):
    for _ in range(n_random_boxes):
        # based on iou, min_size and ratio add new random boxes
        random_box = generate_random_box(overlap_coord=overlap_coord,overlapping_boxes=overlapping_boxes)
        # Append the random box to overlapping_boxes tensor
        overlapping_boxes = torch.cat((overlapping_boxes, random_box.unsqueeze(0)), dim=0)
    return overlapping_boxes

# Test add_n_random_boxes function
def test_add_n_random_boxes():
    # Define overlap_coord as (x1, y1, x2, y2)
    overlap_coord = (10, 10, 100, 100)

    # Define initial overlapping_boxes tensor (4, 4) where each row is a box (x, y, w, h)
    overlapping_boxes = torch.tensor([
        [20, 20, 10, 10],
        [30, 30, 20, 20],
        [40, 40, 15, 15],
        [50, 50, 25, 25]
    ])

    # Number of random boxes to be added
    n_random_boxes = 5

    # Printing initial state
    print("Initial overlapping_boxes tensor:")
    print(overlapping_boxes)

    # Call the add_n_random_boxes function
    result = add_n_random_boxes(overlap_coord, overlapping_boxes, n_random_boxes)

    # Printing final state
    print(f"After adding {n_random_boxes} random boxes:")
    print(result)


# Execute the test function
if __name__ == "__main__":
    test_add_n_random_boxes()