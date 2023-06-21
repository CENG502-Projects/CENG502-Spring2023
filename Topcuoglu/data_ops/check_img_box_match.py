"""Note that aspect ratio is not protected for UniVIP, resize each image."""
from tqdm import tqdm
import pickle
from icecream import ic
import time
from pathlib import Path
import cv2

DEBUG = True
IMAGES_PATH = Path("/raid/utku/datasets/COCO_dataset/unlabeled2017")
check_images_path = Path("check_images")
NUM_IMGS = 1000

check_images_path.mkdir(exist_ok=True)

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

def get_img_paths(num_imgs):
    images_path_list = sorted(IMAGES_PATH.iterdir())
    return images_path_list[:num_imgs]

if __name__ == "__main__":
    # Read the JSON file
    # Load data from pickle file
    
    TARGET_PKL = "/raid/utku/datasets/COCO_dataset/COCO_proposals/final_proposals/unlabeled2017_selective_search_proposal_enumerated_filtered_96_with_names.pkl"
    ic("load_pkl")
    raw_data = load_pkl(TARGET_PKL)
    images_box_dict = raw_data["bbox"]

    for count, (img_path, bbox_list) in tqdm(list(enumerate(images_box_dict.items()))[:NUM_IMGS]):
        # Read the image
        img = cv2.imread(str(IMAGES_PATH/ img_path))
        # resized_img = cv2.resize(img, (640, 640))

        # Draw bounding boxes on the image
        for bbox in bbox_list:
            x, y, width, height = bbox
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the image with bounding boxes
        save_path = Path(img_path).name
        cv2.imwrite(str(check_images_path/save_path), img)

    # cv2.destroyAllWindows()