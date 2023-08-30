"""Example experiment for a linear baseline method."""
import glob
import json
import logging
import os
# set gpu private
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf


import scipy.stats
import numpy as np


from task2_regression.models.linear import simple_linear_model, pearson_loss_cut, pearson_metric_cut
from util.dataset_generator import DataGenerator, create_tf_dataset


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
        logging.info(f"Scores for subject {subject}:")
           # evaluate model
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))


        # metrics = model.metrics_names
        # evaluation[subject] = dict(zip(metrics, results))
    return evaluation



if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    fs = 64
    window_length = 5 * fs  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 1*fs
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False

    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"],config["derivatives_folder"],  config["split_folder"])
    stimulus_features = ["mel"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_linear_baseline")
    os.makedirs(results_folder, exist_ok=True)

    # train a sub dependent model for each sub
    # Create a dataset generator for each training subject
    # Get all different subjects from the training set
    all_subs = list(
        set([os.path.basename(x).split("_-_")[1] for x in glob.glob(os.path.join(data_folder, "train_-_*"))]))


    # create a simple linear model
    model = simple_linear_model(integration_window = int(fs*0.25), nb_filters=10)
    model.summary()
    model_path = os.path.join(results_folder, f"model.h5")
    training_log_filename = f"training_log.csv"
    results_filename = f'eval.json'


    if only_evaluate:
        # load weights
        model.load_weights(model_path)
    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
            workers = tf.data.AUTOTUNE,
            use_multiprocessing=True

        )

    # Evaluate the model on test set
    # Create a dataset generator for each test subject
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        test_generator = DataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size=1, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    # Evaluate the model
    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")
