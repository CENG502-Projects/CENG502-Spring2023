import json
import pickle
from icecream import ic
import time
from pathlib import Path

DEBUG = True
ANNOTATION_INFO = '/home/kuartis-dgx1/utku/UniVIP/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'

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
    # Read the JSON file
    with open(ANNOTATION_INFO) as f:
        annotation_info_data = json.load(f)

    # Create the dictionary mapping indexes to "file_name"
    file_name_dict_json = {index: item['file_name'] for index, item in enumerate(annotation_info_data['images'])}

    TARGET_PKL = "/home/kuartis-dgx1/utku/UniVIP/COCO/unlabeled2017_selective_search_proposal_enumerated_filtered_96.pkl"
    ic("load_pkl")
    raw_data = load_pkl(TARGET_PKL)
    images_box_dict = raw_data["bbox"]
    new_images_box_dict = {
        file_name_dict_json[img_idx]: images_box_dict[img_idx]
        for img_idx in range(len(images_box_dict))
    }
    raw_data["bbox"] = new_images_box_dict

    ic("dump_pkl")
    dump_pkl(f'{Path(TARGET_PKL).stem}_with_names.pkl', raw_data)