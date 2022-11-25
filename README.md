Auditory-eeg-challenge-2023-code
================================
This is the codebase for the [2023 ICASSP Auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2023).
This codebase contains baseline models and code to preprocess stimuli for both tasks.

# Prerequisites

Python >= 3.6

# General setup

Steps to get a working setup:

## 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
# Clone this repository
git clone https://github.com/exporl/auditory-eeg-challenge-2023-code

# Go to the root folder
cd auditory-eeg-challenge-2023-code

# Optional: install a virtual environment
python3 -m venv venv # Optional
source venv/bin/activate # Optional

# Install requirements.txt
python3 -m install requirements.txt
```

## 2. [Download the data](https://kuleuven-my.sharepoint.com/personal/lies_bollens_kuleuven_be/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flies%5Fbollens%5Fkuleuven%5Fbe%2FDocuments%2FICASSP%2D2023%2Deeg%2Ddecoding%2Dchallenge%2Ddataset&ga=1) 

You will need a password, which you will receive when you [register](https://exporl.github.io/auditory-eeg-challenge-2023/registration/).
The folder contains multiple folders (and `zip` files containing the same data as their corresponding folders). For bulk downloading, we recommend using the `zip` files, as there is a bug in Onedrive when it has to zip files >= 4GB

   1. `split_data(.zip)` contains already preprocessed, split and normalized data; ready for model training evalation. 
If you want to get started quickly, you can opt to only download this folder/zipfile

   2. `preprocessed_eeg(.zip)` and `preprocessed_stimuli(.zip)` contain preprocessed EEG and stimuli files (envelope and mel features) respectively.
At this stage data is not yet split into different sets and normalized. To go from this to the data in `split_data`, you will have to run [run_splitting_and_normalization.py](./run_splitting_and_normalization.py).

   3. `raw_eeg(_x.zip)` and `stimuli(.zip)` contain the raw EEG and stimuli files. If you want to process the stimuli files, you can run [run_preprocessing.py](./run_preprocessing.py). 
Currently, no preprocessing code is made available to preprocess EEG, so you will have to write your own implementation or use the precomputed `processed_eeg` folder.

Make sure to download/unzip these folders into the same folder (e.g. `challenge_folder`).

![data_diagram](./images/data_diagram.svg)

## 3. Adjust the [config.json](./config.json) accordingly
Adjust `dataset_folder` in [config.json](./config.json) from `None` to the absolute path to the folder containing all data (The `challenge_folder` from the previous point).
  

OK, you should be all setup now!

    

# Running the tasks

Each task has already some ready-to-go experiments files defined to give you a
baseline and make you acquainted with the problem. The experiment files live
in the `experiment` subfolder for each task. The training log,
best model and evaluation results will be stored in a folder called
`results_{experiment_name}`.

## Task1: Match-mismatch
    
By running [task1_match_mismatch/experiments/dilated_convolutional_model.py](./task1_match_mismatch/experiments/dilated_convolutional_model.py),
you can train the dilated convolutional model introduced by Accou et al. [(2021a)](https://doi.org/10.23919/Eusipco47968.2020.9287417) and [(2021b)](https://doi.org/10.1088/1741-2552/ac33e9)



## Task2: regression (reconstructing envelope from EEG)

By running [task2_regression/experiments/linear_baseline.py](./task2_regression/experiments/linear_baseline.py), you can 
train and evaluate a simple linear baseline model with Pearson correlation as a loss function, as in [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945)

By running [task2_regression/experiments/vlaai.py](./task2_regression/experiments/vlaai.py), you can train/evaluate
the VLAAI model as proposed by [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945). You can find a pre-trained model at [VLAAI's github page](https://github.com/exporl/vlaai).


# Changing preprocessing

You can run the preprocesing for the stimuli using [run_preprocessing.py](./run_preprocessing.py),
and subsequently split and/or normalize the data by running [run_splitting_and_normalization.py](./run_splitting_and_normalization.py).
To see all builtin options, you can run these script with a `--help` flag, e.g.:
```bash
python3 run_preprocessing.py --help
```

You can add and modify the speech feature extraction, splitters and normalizers easily by:

1. Deriving a class from the appropriate base class
   1. `FeatureExtractor` in [util/stimulus_processing/feature_extraction/base.py](./util/stimulus_processing/feature_extraction/base.py) for feature extraction.
   2. `Splitter` in [util/splitters/base.py](./util/splitters/base.py) for splitters
   3. `Normalizer` in for [util/normalizers/base.py](./util/normalizers/base.py) for normalizers

2. Adding the new class to the correct factory function
   1. `speech_feature_factory` in [util/stimulus_processing/factory.py](./util/stimulus_processing/factory.py) for feature extraction.
   2. `Splitter` in [util/splitters/factory.py](./util/splitters/factory.py) for splitters
   3. `Normalizer` in for [util/normalizers/factory.py](./util/normalizers/factory.py) for normalizers

Then you can use your newly defined class with [run_preprocessing.py](./run_preprocessing.py) and
[run_splitting_and_normalization.py](./run_splitting_and_normalization.py).
