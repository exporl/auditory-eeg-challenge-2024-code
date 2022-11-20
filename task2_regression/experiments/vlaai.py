"""Example of vlaai model trained in the regression task."""

import os
import numpy as np
import glob
import json
import tensorflow as tf
from task2_regression.models import vlaai
from task2_regression.util.dataset_generator import create_generator

def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(
            zip(metrics, results)
        )
    return evaluation




if __name__ == '__main__':

    # Preprocessed EEG path
    # The path to the challenge dataset is in /util/dataset_root_dir.json
    os.chdir('..')
    dataset_path_file = os.path.join(os.getcwd(), 'util', 'dataset_root_dir.json')
    with open(dataset_path_file, 'r') as f:
        dataset_root_dir = json.load(f)
    preprocessed_eeg_dir = os.path.join(dataset_root_dir, 'train', 'preprocessed_eeg')
    all_subjects = glob.glob(os.path.join(preprocessed_eeg_dir, 'sub*'))
    # You can use indexing to select a subject of subjects
    # for faster run
    all_subjects = all_subjects[0:]

    # Stimulus path
    envelope_dir = os.path.join(os.getcwd(), 'create_data', 'data_dir', 'train_dir', 'envelope')
    # Change back to current directory
    os.chdir("experiments")

    # Paramters
    window_length = 5*64
    hop_length = 1*32
    batch_size = 32

    # Train, validation, test split: ( 80 %, 10 %, 10%)
    num_of_test_validation_subjects = int(np.ceil(len(all_subjects) * 0.1))
    test_subjects = all_subjects[-num_of_test_validation_subjects:]
    val_subjects = all_subjects[-2*num_of_test_validation_subjects: -num_of_test_validation_subjects]
    train_subjects = all_subjects[: -2 * num_of_test_validation_subjects]

    # Train set
    eeg_files = []
    for subject in train_subjects:
        eeg_files += glob.glob(os.path.join(subject, '*', 'sub*.pkl'))
    dataset_train = create_generator(eeg_files, envelope_dir, window_length, hop_length, batch_size)

    # Validation set
    eeg_files = []
    for subject in val_subjects:
        eeg_files += glob.glob(os.path.join(subject, '*', 'sub*.pkl'))
    dataset_val = create_generator(eeg_files, envelope_dir, window_length, hop_length, batch_size)

    model = vlaai.vlaai()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=vlaai.pearson_metric,
        loss=vlaai.pearson_loss,
    )
    print(model.summary())

    only_evaluate = True
    if only_evaluate:
        model.load_weights("results/vlaai.h5")
    else:

        model.fit(
            dataset_train,
            epochs=2,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    "results/vlaai.h5", save_best_only=True
                ),
                tf.keras.callbacks.CSVLogger("results/training_log.csv"),
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                ),
            ],
        )

    # Evaluate on test set
    # Create a test dataset for each subject
    datasets_test = {}
    for subject in test_subjects:
        eeg_files = glob.glob(os.path.join(subject, '*', 'sub*.pkl'))
        datasets_test[subject] = create_generator(eeg_files, envelope_dir, window_length, hop_length, batch_size)

    # Evaluate the model
    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    with open("results/eval.json", "w") as fp:
        json.dump(evaluation, fp)