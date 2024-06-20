import os
import pickle
import numpy as np
import pandas as pd
from math import floor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

folder_path = 'data_diffs_clean\\'
refactoring_df = pd.read_csv('analysis_results\\refactoring_details_post.csv')
all_present_actions = ['delete-tree', 'insert-node', 'move-tree', 'delete-node', 'update-node', 'insert-tree']
refactorings = refactoring_df.columns[1:]

# Function to count the occurrences of each action in a file
def get_action_list(lines):
    blocks = []
    current_block = []
    actions = []

    for line in lines:
        if line.strip() == '===':
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    for block in blocks:
        action = block[1].strip()
        actions.append(action)
    
    return actions

# Function to parse edit script files and extract sequences of minimal edit actions
def parse_edit_script_sequences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    sequence = get_action_list(content.split('\n'))
    return sequence

# Function to create n-gram features from sequences of actions
def create_ngram_features(sequences, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, ngram_range=ngram_range, token_pattern=None)
    sequence_strs = [' '.join(seq) for seq in sequences]
    X = vectorizer.fit_transform(sequence_strs)
    return X, vectorizer

# Function to prepare data with minimal edit action sequences
def prepare_data_sequences(folder_path, refactoring_df):
    edit_script_sequences = []
    ids = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            sequences = parse_edit_script_sequences(file_path)
            file_id = int(filename.split('_')[0])
            edit_script_sequences.append(sequences)
            ids.append(file_id)

    sequence_df = pd.DataFrame({'id': ids, 'sequences': edit_script_sequences})

    merged_df = pd.merge(sequence_df, refactoring_df, on='id', how='inner')

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

# Function to run experiments with various models, hyperparameters, and train-test splits using sequence features
def run_experiments_sequences(merged_df, refactorings, models, param_grids, test_splits, random_state=17):
    clfs = {}
    cms = {}
    accs = {}
    summary = {}

    for refactoring in refactorings:
        print(f"Running experiments for {refactoring}...")
        target = f'{refactoring}_Label'
        y = merged_df[target]
        print(f"Refactoring type: {refactoring}, Total positive samples: {y.sum()}")

        if y.sum() < 6:
            print(f"Skipping {refactoring} due to low sample count\n\n")
            continue

        for test_size in test_splits:
            y_pos = y[y == True]
            y_neg = y[y == False]
            
            # Ensure samples split
            y_pos_train = y_pos.sample(n=floor(len(y_pos)*(1-test_size)), random_state=random_state)
            
            if len(y_neg) > 2*len(y_pos_train):
                y_neg_train = y_neg.sample(n=2*len(y_pos_train), random_state=random_state)
            else:
                y_neg_train = y_neg.sample(n=floor(len(y_neg)/2), random_state=random_state)

            y_pos_test = y_pos[~y_pos.index.isin(y_pos_train.index)]
            y_neg_test = y_neg[~y_neg.index.isin(y_neg_train.index)]

            y_train = y.loc[y_pos_train.index.union(y_neg_train.index)]
            y_test = y.loc[y_pos_test.index.union(y_neg_test.index)]

            X_train_seq = merged_df.loc[y_train.index, 'sequences']
            X_test_seq = merged_df.loc[y_test.index, 'sequences']

            X_train, vectorizer = create_ngram_features(X_train_seq)
            X_test = vectorizer.transform([' '.join(seq) for seq in X_test_seq])

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
    merged_df = prepare_data_sequences(folder_path, refactoring_df)
    print('Data prepared')

    models = {
        'RandomForest': RandomForestClassifier(random_state=17),
        'SVM': SVC(random_state=17),
        'LogisticRegression': LogisticRegression(random_state=17)
    }

    param_grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'LogisticRegression': {'C': [0.1, 1, 10], 'max_iter': [100, 200, 300]}
    }

    test_splits = [0.2, 0.3, 0.4, 0.5]

    clfs, cms, accs, summary = run_experiments_sequences(merged_df, refactorings, models, param_grids, test_splits)

    with open('cluster_results\\classification_accuracy_MEA_split.txt', 'w') as f:
        f.write(str(accs))
    with open('cluster_results\\classification_summary_MEA_split.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print_summary(accs, cms)