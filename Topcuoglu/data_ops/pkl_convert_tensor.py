import pickle
from icecream import ic
import time
from pathlib import Path
import torch

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

# Check if the boxes correspond to correct images (sorted)
def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

def dump_pkl(pkl_file, data):
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    TARGET_PKL = "/raid/utku/datasets/COCO_dataset/COCO_proposals/final_proposals/unlabeled2017_selective_search_proposal_enumerated_filtered_96_with_names.pkl"
    ic("load_pkl")
    raw_data = load_pkl(TARGET_PKL)
    images_box_dict = raw_data["bbox"]
    for img_name, box_list in images_box_dict.items():
        images_box_dict[img_name] = torch.tensor(box_list)
    raw_data["bbox"] = images_box_dict
    ic("dump_pkl")
    dump_pkl(f'{Path(TARGET_PKL).stem}_with_tensors.pkl', raw_data)