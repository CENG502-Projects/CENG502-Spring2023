"""Takes approximately
5 minutes 51 seconds for train.
6 minutes 10 seconds for unlabelled"""

import json
import pickle
from pathlib import Path
import time

file_path = '/home/kuartis-dgx1/utku/UniVIP/COCO/train2017_selective_search_proposal.json'

# Start time
start_time = time.time()

# Load JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Save data as pickle
with open(f'{Path(file_path).stem}_temp.pkl', 'wb') as f:
    pickle.dump(data, f)


# Calculate elapsed time
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Elapsed time: {minutes} minutes {seconds} seconds")