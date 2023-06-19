from tqdm import tqdm
import pickle
from icecream import ic
import time
from pathlib import Path
import cv2

DEBUG = True
check_images_path = Path("check_images")

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


PICKLE_FILE = "/home/kuartis-dgx1/utku/UniVIP/COCO/unlabeled2017_selective_search_proposal.pkl"
# Load the pickle file
ic("load")
with open(PICKLE_FILE, 'rb') as file:
    data = pickle.load(file)

bbox = data['bbox']

bbox_dict = dict(enumerate(bbox))
# Update the data dictionary with the new bbox_dict
data['bbox'] = bbox_dict
print(data)

ic("dump")
# Save the updated data dictionary as a new pickle file
with open(f'{Path(PICKLE_FILE).stem}_enumerated.pkl', 'wb') as file:
    pickle.dump(data, file)
ic("dumped")
