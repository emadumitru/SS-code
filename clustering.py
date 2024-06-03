from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(data_folder, refactoring_details_path):
    ref_details = pd.read_csv(refactoring_details_path)
    train_documents = []
    test_documents = []
    train_ids = []
    test_ids = []
    training_labels = {}
    testing_labels = {}

    # Initialize training labels for all refactoring types
    ref_types = [d.replace('_single', '') for d in os.listdir(data_folder) if '_single' in d]
    for ref_type in ref_types:
        training_labels[ref_type] = []
        testing_labels[ref_type] = []

    # Load training data from _single directories and assign labels
    for ref_type in os.listdir(data_folder):
        if '_single' in ref_type:
            ref_type_clean = ref_type.replace('_single', '')
            subfolder_path = os.path.join(data_folder, ref_type)
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.md'):
                    file_path = os.path.join(subfolder_path, filename)
                    with open(file_path, 'r') as file:
                        content = file.read()
                        train_documents.append(content)
                        train_ids.append(filename.split('_')[0])
                        # Initialize labels for all refactoring types as 0
                        for key in training_labels:
                            training_labels[key].append(0)
                        # Set the current refactoring type label to 1
                        training_labels[ref_type_clean][-1] = 1

    # Load testing data from _mixed directories and assign labels based on CSV
    for ref_type in os.listdir(data_folder):
        if '_mixed' in ref_type:
            ref_type_clean = ref_type.replace('_mixed', '')
            subfolder_path = os.path.join(data_folder, ref_type)
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.md'):
                    file_path = os.path.join(subfolder_path, filename)
                    with open(file_path, 'r') as file:
                        content = file.read()
                        test_documents.append(content)
                        file_id = filename.split('_')[0]
                        test_ids.append(file_id) 
                        # Initialize labels for all refactoring types as 0
                        for key in testing_labels:
                            testing_labels[key].append(0)
                        # Set labels based on occurrence counts from the CSV
                        for key in ref_types:
                            count = ref_details.loc[ref_details['id'] == int(file_id), key].values[0]
                            if count > 0:
                                testing_labels[key][-1] = 1

    return train_documents, train_ids, test_documents, test_ids, training_labels, testing_labels

def train_and_evaluate(train_docs, train_labels, test_docs, test_labels, path_save='analysis_results\\'):
    vectorizer=CountVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    classifiers = {}
    results = {}
    summary = {}
    
    for refactoring_type, labels in train_labels.items():
        clf = RandomForestClassifier()
        clf.fit(X_train, labels)
        classifiers[refactoring_type] = clf

        # Evaluate classifier
        predictions = clf.predict(X_test)
        true = np.array(test_labels[refactoring_type])
        cm = confusion_matrix(true, predictions)
        acc = accuracy_score(true, predictions)
        
        # Store results
        results[refactoring_type] = {
            'confusion_matrix': cm,
            'accuracy': acc
        }
        summary[refactoring_type] = [acc, *cm.ravel()]  # Flatten confusion matrix and prepend accuracy

    # Serialize results and summary
    with open(path_save + 'classification_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    with open(path_save + 'classification_results.txt', 'w') as f:
        f.write(str(results))
    
    return classifiers, results

data_folder = "dataset_clean\\"
refactoring_details_path = 'analysis_results\\refactoring_details_post.csv'
train_docs, train_ids, test_docs, test_ids, train_labels, test_labels = load_data(data_folder, refactoring_details_path)
classifiers, results = train_and_evaluate(train_docs, train_labels, test_docs, test_labels)

print("Classification results saved successfully.")

print("Accuracy  and TN/TP/FN/FP for each refactoring type:")
for refactoring_type, result in results.items():
    print(f"{refactoring_type}: {round(result['accuracy'],3)} -- TN/TP/FN/FP : {result['confusion_matrix'][0][0]}/{result['confusion_matrix'][1][1]}/{result['confusion_matrix'][1][0]}/{result['confusion_matrix'][0][1]}")