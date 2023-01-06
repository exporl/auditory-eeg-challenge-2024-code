"""
Sample code to generate test labels (reconstructed envelopes) for
the regression task. The requested format for submitting the reconstructed envelopes is
as follows:
for each subject a json file containing a python dictionary in the
format of  ==> {'sample_id': reconstructed_envelope, ... }.
"""

import os
import glob
import json
import numpy as np
from task2_regression.models.linear import simple_linear_model


if __name__ == '__main__':

    # Parameters
    window_length = 60*64  # one minute
    # Root dataset directory containing test set
    # Change the path to the downloaded test dataset dir
    dataset_dir = 'path/to/test/dataset'
    # Path to your pretrained model
    pretrained_model = os.path.join(os.path.dirname(__file__), 'results_linear_baseline', 'model.h5')

    # Define and load the pretrained model
    model = simple_linear_model()
    model.load_weights(pretrained_model)

    test_data = glob.glob(os.path.join(dataset_dir, 'sub*.json'))
    for sub_path in test_data:
        subject = os.path.basename(sub_path).split('.')[0]

        with open(sub_path, 'r') as f:
            sub_dictionary = json.load(f)

        # Get test data from subject dictionary
        id_list, sub_data = zip(*sub_dictionary.items())

        # Normalize data
        data_mean = np.expand_dims(np.mean(sub_data, axis=1), axis=1)
        data_std = np.expand_dims(np.std(sub_data, axis=1), axis=1)
        sub_data = (sub_data - data_mean) / data_std

        predictions = model.predict(sub_data)
        # Make predictions json-serializable
        predictions = [np.array(value).tolist() for value in np.squeeze(predictions)]

        # Create dictionary from id_list and predictions
        sub = dict(zip(id_list, predictions))

        prediction_dir = os.path.join(os.path.dirname(__file__), 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)




