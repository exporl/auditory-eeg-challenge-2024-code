Auditory-eeg-challenge-2024-code
================================
This is the codebase for the [2024 ICASSP Auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2024).
This codebase contains baseline models and code to preprocess stimuli for both tasks.

# Prerequisites

Python >= 3.6

# General setup

Steps to get a working setup:

## 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
# Clone this repository
git clone https://github.com/exporl/auditory-eeg-challenge-2024-code

# Go to the root folder
cd auditory-eeg-challenge-2024-code

# Optional: install a virtual environment
python3 -m venv venv # Optional
source venv/bin/activate # Optional

# Install requirements.txt
python3 -m install requirements.txt
```

## 2. [Download the data](https://homes.esat.kuleuven.be/~lbollens/)

You will need a password, which you will receive when you [register](https://exporl.github.io/auditory-eeg-challenge-2024/registration/).
The folder contains multiple folders (and `zip` files containing the same data as their corresponding folders). For bulk downloading, we recommend using the `zip` files, 

   1. `split_data(.zip)` contains already preprocessed, split and normalized data; ready for model training/evaluation. 
If you want to get started quickly, you can opt to only download this folder/zipfile.

   2. `preprocessed_eeg(.zip)` and `preprocessed_stimuli(.zip)` contain preprocessed EEG and stimuli files (envelope and mel features) respectively.
At this stage data is not yet split into different sets and normalized. To go from this to the data in `split_data`, you will have to run the `split_and_normalize.py` script ([preprocessing_code/split_and_normalize.py](./preprocessing_code/split_and_normalize.py) )

   3. `sub_*(.zip)` and `stimuli(.zip)` contain the raw EEG and stimuli files. 
If you want to recreate the preprocessing steps, you will need to download these files and then run `sparrKULee.py` [(preprocessing_code/sparrKULee.py)](./preprocessing_code/sparrKULee.py) to preprocess the EEG and stimuli and then run the `split_and_normalize.py` script to split and normalize the data.
It is possible to adapt the preprocessing steps in `sparrKULee.py` to your own needs, by adding/removing preprocessing steps. For more detailed information on the pipeline, see the [brain_pipe documentation](https://exporl.github.io/brain_pipe/).


Note that it is possible to use the same preprocessed (and split) dataset for both task 1 and task 2, but it is not required.



## 3. Adjust the `config.json` accordingly

There is a general `config.json` defining the folder names and structure for the data (i.e. [util/config.json](./util/config.json) ).
Adjust `dataset_folder` in the `config.json` file from `null` to the absolute path to the folder containing all data (The `challenge_folder` from the previous point). 
If you follow the BIDS structure, by downloading the whole dataset, the folders preprocessed_eeg, preprocessed_stimuli and split_data, should be located inside the 'derivatives' folder. If you only download these three folders, make sure they are either in a subfolder 'derivatives', or change the 'derivatives' folder in the config, otherwise you will get a file-not-found error when trying to run the experiments. 
  

OK, you should be all setup now!

    

# Running the tasks

Each task has already some ready-to-go experiments files defined to give you a
baseline and make you acquainted with the problem. The experiment files live
in the `experiment` subfolder for each task. The training log,
best model and evaluation results will be stored in a folder called
`results_{experiment_name}`. For general ideas, you might want to look at the winners of the 
[previous ICASSP auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2023).  

## Task1: Match-mismatch
    
By running [task1_match_mismatch/experiments/dilated_convolutional_model.py](./task1_match_mismatch/experiments/dilated_convolutional_model.py),
you can train the dilated convolutional model introduced by Accou et al. [(2021a)](https://doi.org/10.23919/Eusipco47968.2020.9287417) and [(2021b)](https://doi.org/10.1088/1741-2552/ac33e9).

Other models you might find interesting are [Decheveigné et al (2021)](https://www.sciencedirect.com/science/article/pii/S1053811918300338), [Monesi et al. (2020)](https://ieeexplore.ieee.org/abstract/document/9054000), [Monesi et al. (2021)](https://arxiv.org/abs/2106.09622),….



## Task2: Regression (reconstructing spectrogram from EEG)

By running [task2_regression/experiments/linear_baseline.py](./task2_regression/experiments/linear_baseline.py), you can 
train and evaluate a simple linear baseline model with Pearson correlation as a loss function, similar to the baseline model used in [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945).

By running [task2_regression/experiments/vlaai.py](./task2_regression/experiments/vlaai.py), you can train/evaluate
the VLAAI model as proposed by [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945). You can find a pre-trained model at [VLAAI's github page](https://github.com/exporl/vlaai).

Other models you might find interesting are: [Thornton et al. (2022)](https://iopscience.iop.org/article/10.1088/1741-2552/ac7976),...
