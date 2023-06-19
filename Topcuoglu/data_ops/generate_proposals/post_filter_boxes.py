""" get_max_iou and box_filter methods were copied from the 
ORL (Unsupervised Object-Level Representation Learning from Scene Images) paper's github repo:
https://github.com/Jiahao000/ORL/tree/2ad64f7389d20cb1d955792aabbe806a7097e6fb 

Minutes passed.
30:42 for train
49:48 for unlabelled dataset
"""

import numpy as np
from tqdm import tqdm
import pickle
from icecream import ic
import time
from pathlib import Path
from multiprocessing import Pool

MP = True
NUM_PROC = 80
DEBUG = True

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


def get_max_iou(filtered_boxes, candidate_box):
    """
    filtered_boxes : multiple coordinate for filtered bounding boxes (x, y, w, h)
    candidate_box :  the coordinate for ground truth bounding box (x, y, w, h)
    return :   the max iou score about filtered_boxes and candidate_box
    """
    # 1.get the coordinate of inters
    ixmin = np.maximum(filtered_boxes[:, 0], candidate_box[0])
    ixmax = np.minimum(filtered_boxes[:, 0] + filtered_boxes[:, 2], candidate_box[0] + candidate_box[2])
    iymin = np.maximum(filtered_boxes[:, 1], candidate_box[1])
    iymax = np.minimum(filtered_boxes[:, 1] + filtered_boxes[:, 3], candidate_box[1] + candidate_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (filtered_boxes[:, 2] * filtered_boxes[:, 3] + candidate_box[2] * candidate_box[3] - inters)

    # 4. calculate the overlaps and find the max overlap between filtered_boxes and candidate_box
    iou = inters / uni
    return np.max(iou)


def box_filter(candidate_boxes, min_size=None, max_ratio=None, max_iou_thr=None):
    """Since the order is important cannot apply async multiprocessing."""
    # NOTE I have to re-generate all candidate_boxes, called filtered_box_proposals.append(box) before max_iou_thr: 
    filtered_box_proposals = []

    for box in candidate_boxes:
        # Calculate width and height of the box
        w, h = box[2], box[3]

        # Filter for size
        if min_size:
            if w < min_size or h < min_size:
                continue

        # Filter for box ratio
        if max_ratio:
            if w / h > max_ratio or h / w > max_ratio:
                continue

        # UTKU ADDED Filter for overlap
        if max_iou_thr:
            if len(filtered_box_proposals):
                iou_max = get_max_iou(np.array(filtered_box_proposals), np.array(box))
                if iou_max > max_iou_thr:
                    continue

        filtered_box_proposals.append(box)
    return filtered_box_proposals # [[x,y,w,h], [x,y,w,h], [x,y,w,h] ...]

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

def filter_proposals_single(boxes_per_image_list):
    # Empty lists will be appended for images without proposals.
    filtered_boxes_64_single = box_filter(candidate_boxes=boxes_per_image_list, min_size=64, max_ratio=3, max_iou_thr=0.5)
    filtered_boxes_96_single = box_filter(candidate_boxes=boxes_per_image_list, min_size=96, max_ratio=3, max_iou_thr=0.5)
    return filtered_boxes_64_single, filtered_boxes_96_single

def filter_proposals_with_indexes_single(args:list):
    idx, boxes_per_image_list = args
    result = filter_proposals_single(boxes_per_image_list)
    return idx, result

def filter_proposals(images_lists):
    filtered_boxes_64 = []
    filtered_boxes_96 = []
    for boxes_per_image_list in tqdm(images_lists):
        filtered_boxes_64_single, filtered_boxes_96_single = filter_proposals_single(boxes_per_image_list)
        # Empty lists will be appended for images without proposals.
        filtered_boxes_64.append(filtered_boxes_64_single)
        filtered_boxes_96.append(filtered_boxes_96_single)
    filtered_data_64 = {"bbox":filtered_boxes_64}
    filtered_data_96 = {"bbox":filtered_boxes_96}
    return filtered_data_64, filtered_data_96

def dump_pkl(pkl_file, data):
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)


def main(images_dicts):
    if MP:
        images_lists_with_indexes = list(images_dicts.items())
        with Pool(processes=NUM_PROC) as p:
            results = list(tqdm(p.imap(filter_proposals_with_indexes_single, images_lists_with_indexes), total=len(images_lists_with_indexes)))
        
        # Sort the results by the original index and extract the results
        results.sort(key=lambda x: x[0])
        filtered_data_64 = {"bbox": {idx: res[0] for idx, res in results}}
        filtered_data_96 = {"bbox": {idx: res[1] for idx, res in results}}
    else:
        filtered_data_64, filtered_data_96 = filter_proposals(images_lists=images_dicts.values())
    return filtered_data_64, filtered_data_96

if __name__ == "__main__":
    # Load data from pickle file
    TARGET_PKL = "/home/kuartis-dgx1/utku/UniVIP/COCO/unlabeled2017_selective_search_proposal_enumerated.pkl"
    ic("load_pkl")
    raw_data = load_pkl(TARGET_PKL)
    # Access the dictionary
    images_dicts = raw_data["bbox"]
    # Create an iterable that includes the original indexes
    ic("filter_proposals")
    filtered_data_64, filtered_data_96 = main(images_dicts=images_dicts)
    
    # Save filtered data into separate JSON files
    ic("dump_pkl")
    dump_pkl(f'{Path(TARGET_PKL).stem}_filtered_64.pkl', filtered_data_64)
    ic("dump_pkl")
    dump_pkl(f'{Path(TARGET_PKL).stem}_filtered_96.pkl', filtered_data_96)
