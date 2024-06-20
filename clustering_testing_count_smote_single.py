import os
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#repress user warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

folder_path = 'data_diffs_clean\\'
refactoring_df = pd.read_csv('analysis_results\\refactoring_details_post.csv')
all_present_actions = ['delete-tree', 'insert-node', 'move-tree', 'delete-node', 'update-node', 'insert-tree']
refactorings = refactoring_df.columns[1:]

# Function to count the occurrences of each action in a file
def count_file_content(lines, actions):
    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == '===':
            if current_block != []:
                blocks.append(current_block)
                current_block = []
        current_block.append(line)

    for block in blocks:
        action = block[1].strip()
        if action in actions:
            actions[action] += 1
    
    return actions

# Function to parse edit script files and count edit actions
def parse_edit_script(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    actions = {action: 0 for action in all_present_actions}
    action_counts = count_file_content(content.split('\n'), actions)
    return action_counts

# Function to create a dataframe with action counts
def create_action_counts_dataframe(folder_path):
    edit_script_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            action_counts = parse_edit_script(file_path)
            edit_script_data.append({'id': int(filename.split('_')[0]), **action_counts})
    df = pd.DataFrame(edit_script_data).fillna(0)
    return df

# Function to merge dataframes and create labels
def prepare_data(folder_path, refactoring_df):
    edit_script_df = create_action_counts_dataframe(folder_path)
    merged_df = pd.merge(edit_script_df, refactoring_df, on='id')
    for refactoring in refactorings:
        merged_df[f'{refactoring}_Label'] = merged_df[refactoring] > 0
    return merged_df

# Function to train and evaluate models
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid=None):
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = model.get_params()
    
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = balanced_accuracy_score(y_test, y_pred)
    
    if cm.shape == (1, 1):
        cm = np.array([[cm[0][0], 0], [0, 0]])
    
    return best_model, cm, acc, best_params

def transform_cm_in_relative_values(cm):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total_true = tp + fn
    total_false = tn + fp
    tn = round(tn / total_false * 100) if total_false > 0 else 0
    tp = round(tp / total_true * 100) if total_true > 0 else 0
    fn = round(fn / total_true * 100) if total_true > 0 else 0
    fp = round(fp / total_false * 100) if total_false > 0 else 0
    return tn, tp, fn, fp

# Function to run experiments with various models and hyperparameters
def run_experiments(merged_df, refactorings, models, param_grids, test_splits, random_state=17):
    clfs = {}
    cms = {}
    accs = {}
    summary = {}

    # keep only where files have max in label type
    merged_df['c_labels'] = sum([merged_df[f'{ref}_Label'] for ref in refactorings])
    merged_df = merged_df[merged_df['c_labels'] == 1]
    
    for refactoring in refactorings:
        print(f"Running experiments for {refactoring}...")
        features = all_present_actions
        target = f'{refactoring}_Label'

        X = merged_df[features]
        y = merged_df[target]
        print(f"Refactoring type: {refactoring}, Total positive samples: {y.sum()}")

        if y.sum() < 10:
            print(f"Skipping {refactoring} due to low sample count\n\n")
            continue

        for test_size in test_splits:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            y_pos_train = y_train[y_train == True]
            y_neg_train = y_train[y_train == False]
            
            print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
            print(f"Positive samples in test set: {y_test.sum()} - train set: {y_train.sum()}")

            for model_name, model in models.items():
                param_grid = param_grids.get(model_name, None)
                clf, cm, acc, best_params = train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid)
                key = f'{refactoring}_{model_name}_{test_size}_{len(y_pos_train)}_{len(y_neg_train)}_{best_params}'
                clfs[key] = clf
                cms[key] = cm
                accs[key] = acc
                summary[key] = [acc] + list(transform_cm_in_relative_values(cm))
        print(f"Experiments for {refactoring} completed\n\n")

    return clfs, cms, accs, summary

# Function to print summary
def print_summary(accs, cms):
    print("\nAccuracy and TN/TP/FN/FP (percentage out of total true and negative) for each refactoring type:")
    summary_data = []
    for ref in accs:
        accuracy = round(accs[ref], 3)
        tn, tp, fn, fp = transform_cm_in_relative_values(cms[ref])
        refactoring_type, model_params = ref.split('_', 1)
        summary_data.append([refactoring_type, model_params, accuracy, tn, tp, fn, fp])
    
    summary_df = pd.DataFrame(summary_data, columns=['Refactoring Type', 'Model and Params', 'Accuracy', 'TN', 'TP', 'FN', 'FP'])
    print(summary_df.to_string(index=False))

# Main script
if __name__ == "__main__":
    merged_df = prepare_data(folder_path, refactoring_df)
    print('Data prepared')

    models = {
        'RandomForest': RandomForestClassifier(random_state=17),
        'SVM': SVC(random_state=17),
        'LogisticRegression': LogisticRegression(random_state=17)
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'max_iter': [100, 200, 300]
        }
    }

    param_grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'LogisticRegression': {'C': [0.1, 1, 10], 'max_iter': [100, 200, 300]}
    }

    test_splits = [0.2, 0.3, 0.4, 0.5]

    clfs, cms, accs, summary = run_experiments(merged_df, refactorings, models, param_grids, test_splits)

    with open('cluster_results\\classification_accuracy_count_smote_single.txt', 'w') as f:
        f.write(str(accs))
    with open('cluster_results\\classification_summary_count_smote_single.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print_summary(accs, cms)
