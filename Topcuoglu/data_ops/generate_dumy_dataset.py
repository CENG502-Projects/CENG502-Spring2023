# Load data from pickle file

import torch
from torch import Tensor
from tqdm import tqdm
import pickle
from icecream import ic
import time
from pathlib import Path
DEBUG = True
TARGET_PKL = "/raid/utku/datasets/COCO_dataset/COCO_proposals/final_proposals/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou.pkl"
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
def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

def dump_pkl(pkl_file, data):
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

ic("load_pkl")
raw_data = load_pkl(TARGET_PKL)
# Access the dictionary
images_dicts = raw_data["bbox"]

# Save only the first 1000 samples to a new dictionary
selected_samples = dict(list(images_dicts.items())[:250])

# Create a new dictionary to store the selected samples
selected_data = {"bbox": selected_samples}

# Save the selected data into a new pickle file
ic("dump_pkl")
dump_pkl(f'{Path(TARGET_PKL).stem}_trial_250.pkl', selected_data)
