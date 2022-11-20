# Creates speech feature: envelope
import glob
import os
import numpy as np
import json
from task1_match_mismatch.util import envelope

# Directory to save the speech features
current_dir = os.getcwd()
save_cache_dir = os.path.join(current_dir, 'data_dir/train_dir/envelope')
os.makedirs(save_cache_dir, exist_ok=True)

# Source speech directory containing raw speech
# The path to the challenge dataset is in /util/dataset_root_dir.json
os.chdir('..')
dataset_path_file = os.path.join(os.getcwd(), 'util', 'dataset_root_dir.json')
os.chdir('create_data')
with open(dataset_path_file, 'r') as f:
    dataset_root_dir = json.load(f)
source_speech_dir = os.path.join(dataset_root_dir, 'train', 'stimuli')
speech_files = glob.glob(os.path.join(source_speech_dir, '*.npz'))

for file in speech_files:
    # Loop over speech files and create envelope
    # and save them
    stimulus = file.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_cache_dir, stimulus + '.npz')

    # If the cache already exists then skip
    if not os.path.isfile(save_path):
        print(file)
        env = envelope.calculate_envelope(file)
        np.savez(save_path, env=env)



