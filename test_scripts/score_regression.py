import os
import glob

import scipy.stats
import json
import numpy as np
from zipfile import ZipFile

subjects_training = ['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-031', 'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-036', 'sub-037', 'sub-038', 'sub-039', 'sub-040', 'sub-041', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-051', 'sub-052', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-057', 'sub-058', 'sub-059', 'sub-060', 'sub-061', 'sub-062', 'sub-063', 'sub-064', 'sub-065', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-071']
subjects_heldout = ['sub-072', 'sub-073', 'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-083', 'sub-081', 'sub-082', 'sub-084', 'sub-085']



path_labels_match_mismatch = os.path.join(os.path.dirname(__file__), 'labels_regression')
path_save_scores = os.path.join(os.path.dirname(__file__), 'results_task_regression')
predicted_results_folder = "PATH_TO_YOUR_RESULTS_FOLDER"


def score_task(predicted_labels):
    # create dictionary for final score, per subject
    score_per_subject = {}
    # load the true labels subject per subject
    for file in glob.glob(os.path.join(path_labels_match_mismatch, '*.json')):
        with open(file, 'r') as f:
            labels = json.load(f)
            subject = os.path.splitext(os.path.basename(file))[0]
            score = []
            for key, label in labels.items():
                # check if key in predicted_labels, else is wrong
                label = label[0]
                if key in predicted_labels:
                    predicted_env = predicted_labels[key]
                    # check if predicted_env is a list
                    if not isinstance(predicted_env, list):
                        print(predicted_env, key, group_name)
                    if len(predicted_env) < len(label):
                        # pad with zeros
                        predicted_env = np.pad(predicted_env, (0, len(label) - len(predicted_env)), 'constant')

                    elif len(predicted_env) > len(label):
                        # truncate
                        predicted_env = predicted_env[:len(label)]
                    score.append(scipy.stats.pearsonr(predicted_env, np.squeeze(label))[0])
                else:
                    score.append(0)

            # print(f'{subject}: {np.mean(score)}')
            score_per_subject[subject] = np.mean(score)

    # print the average score for subjects in the subjects_training
    scores_training = [score_per_subject[x] for x in subjects_training]
    scores_test = [score_per_subject[x] for x in subjects_heldout]

    total_score = 2/3*np.mean(scores_training)+ 1/3*np.mean(scores_test)

    print(f'Total score : {np.mean(total_score)}')
    print(f'Total score training : {np.mean(scores_training)}')
    print(f'Total score test : {np.mean(scores_test)}')
    score = {'within': list(sorted(scores_training)), 'heldout': list(sorted(scores_test)), 'total': total_score}

    return score




all_zips = glob.glob(predicted_results_folder + '/*.zip')
for zip_file in all_zips:
    # unzip file
    with ZipFile(zip_file, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        group_folder = os.path.join(predicted_results_folder, zip_file.split('/')[-1].split('.')[0].replace(' ', ''))
        if not os.path.exists(group_folder):
            os.mkdir(group_folder)
            zipObj.extractall(group_folder)


all_scores = {}
# Loop over all the unzipped folders
for group_results in glob.glob(os.path.join(predicted_results_folder, '*')):
    if os.path.isdir(group_results):
        # Loop over all the json files in the folder
        predicted_labels = {}
        # find all json files in group_results and subfolders
        for file in glob.glob(os.path.join(group_results, '**/*.json'), recursive=True):
            with open(file, 'r') as f:
                temp_labels = json.load(f)
                predicted_labels.update(temp_labels)
                # predicted_labels.update(json.load(f))

        group_name = group_results.split('/')[-1].split('group_')[-1].split('_sub')[0]

        score = score_task(predicted_labels)
        print(group_name, score['total'])
        submission_number = int(group_results.split('.')[0].split('submission_')[-1])
        if group_name not in all_scores:
            all_scores[group_name] = {}
        all_scores[group_name][submission_number] = score

last_scores = {}
for group_name, results  in all_scores.items():
    last_submission = max(results.keys())
    last_scores[group_name] = results[last_submission]


# save last_scores to json file
with open(os.path.join(path_save_scores, 'scores_task2.json'), 'w') as f:
    json.dump(last_scores, f)

mean_variances = []
for group_name, results  in all_scores.items():
    for submission_number, scores in results.items():
        variances_group = {'group_name': group_name,'submission':submission_number,'within_mean': np.mean(scores['within']), 'within_std': np.std(scores['within']),'heldout_mean': np.mean(scores['heldout']),'heldout_std': np.std(scores['heldout']), 'total':scores['total']}
        mean_variances.append(variances_group)


mean_variances = list(sorted(mean_variances, key=lambda x: x['total'], reverse=True))

with open(os.path.join( path_save_scores,
        'scores_task2_means.json'), 'w') as f:
    json.dump(mean_variances, f)








