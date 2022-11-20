"""Example experiment for dilation model."""
import argparse
import json
import os
import tensorflow as tf

from task1_match_mismatch.util.dataset_generator import create_generator
from task1_match_mismatch.models.baseline import dilation_model


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



if __name__ == "__main__":

    # Provide the path of the dataset
    # which is split already to train, val, test
    os.chdir("..")
    dataset_dir = os.path.join(os.getcwd(), 'create_data', 'data_dir', 'train_dir')
    # Change back to current directory
    os.chdir("experiments")

    # Parameters
    # Length of the decision window
    window_length = 3 * 64  # 3 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64
    # Number of samples (space) between end of matched speech and beginning of mismatched speech
    spacing = 64
    batch_size = 32

    only_evaluate = False

    # Create a directory to store (intermediate) results
    os.makedirs("results", exist_ok=True)

    #create dilation model
    model = dilation_model(time_window=window_length)

    if only_evaluate:
        model = tf.keras.models.load_model("results/model.h5")
    else:

        dataset_dir_train = os.path.join(dataset_dir, "train")
        # Create list of numpy array files
        files = [os.path.join(dataset_dir_train, f) for f in os.listdir(dataset_dir_train)]
        dataset = create_generator(files, window_length, hop_length, spacing, batch_size)

        # Create the generator for the validation set
        dataset_dir_val = os.path.join(dataset_dir, "val")
        files_val = [os.path.join(dataset_dir_val, f) for f in os.listdir(dataset_dir_val)]
        dataset_val = create_generator(files_val, window_length, hop_length, spacing, batch_size)

        # Train the model
        model.fit(
            dataset,
            epochs=20,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    "results/model.h5", save_best_only=True
                ),
                tf.keras.callbacks.CSVLogger("results/training_log.csv"),
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                ),
            ],
        )


    # Evaluate the model on test set

    # Create a dataset generator for each test subject
    dataset_dir_test = os.path.join(dataset_dir, "test")
    files_test = [os.path.join(dataset_dir_test, f) for f in os.listdir(dataset_dir_test)]
    # Get all different subjects from the test set
    subjects = [f.split("_")[0] for f in os.listdir(dataset_dir_test)]
    subjects = list(set(subjects))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in files_test if sub in f]
        datasets_test[sub] = create_generator(files_test_sub, window_length, spacing, batch_size)

    # Evaluate the model
    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    with open("results/eval.json", "w") as fp:
        json.dump(evaluation, fp)
