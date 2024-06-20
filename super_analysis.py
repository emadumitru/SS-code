import pandas as pd
import pickle
import os

def load_summaries(file_paths):
    summaries = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f'File {path} does not exist')
            continue
        with open(path, 'rb') as f:
            summaries.append((path, pickle.load(f)))
    return summaries

def extract_all_results(summaries):
    data = []
    for path, summary in summaries:
        file = path.split('classification_summary_')[1].split('.pkl')[0]
        for ref_type, metrics in summary.items():
            accuracy, tn, tp, fn, fp = metrics
            split = ref_type.split('_', 1)
            ref_type, model_params = split[0], split[1]
            model, split, pos_train, neg_train = model_params.split('_')[:4]
            model_params = model_params.split('_')[4:]
            data.append([ref_type, file, model, split, pos_train, neg_train, accuracy, tn, tp, fn, fp, model_params])

    df_data = pd.DataFrame(data, columns=['Refactoring Type', 'Source File', 'Model', 'Split', 'Pos Train', 'Neg Train', 'Accuracy', 'TN', 'TP', 'FN', 'FP', 'Model Params'])
    df_data['Approach'] = df_data['Source File'].apply(lambda x: x.split('_')[0])
    df_data['Type Split'] = df_data['Source File'].apply(lambda x: True if x.find('split') != -1 else False)
    df_data['Type File'] = df_data['Source File'].apply(lambda x: True if x.find('single') != -1 else False)
    return df_data

def get_max_performance(df):
    # keep only best accuracy, best TN and best TP for comination file and ref type
    best_accuracy = df.loc[df.groupby(['Source File', 'Refactoring Type'])['Accuracy'].idxmax()]
    best_tn = df.loc[df.groupby(['Source File', 'Refactoring Type'])['TN'].idxmax()]
    best_tp = df.loc[df.groupby(['Source File', 'Refactoring Type'])['TP'].idxmax()]
    best_accuracy["metric"] = "Accuracy"
    best_tn["metric"] = "TN"
    best_tp["metric"] = "TP"
    return best_accuracy, best_tn, best_tp

def round_all_values(df):
    for col in df.columns:
        if col in ['Accuracy', 'TN', 'TP', 'FN', 'FP']:
            df[col] = df[col].round(3)
    return df

def create_aggregate_tables(summary_df):
    # Comprehensive Best Results Table
    best_results = get_max_performance(summary_df)
    best_results = pd.concat(best_results).sort_values(['Source File', 'Refactoring Type']).reset_index(drop=True)

    # Average Model Performance per Label Table
    average_model_performance = summary_df.groupby(['Source File', 'Refactoring Type', 'Model']).agg({
        'Accuracy': 'mean',
        'TN': 'mean',
        'TP': 'mean'
    }).reset_index()

    # Average Test Split Performance per Label Table
    average_test_split_performance = summary_df.groupby(['Source File', 'Refactoring Type', 'Split']).agg({
        'Accuracy': 'mean',
        'TN': 'mean',
        'TP': 'mean'
    }).reset_index()

    best_results = round_all_values(best_results)
    average_model_performance = round_all_values(average_model_performance)
    average_test_split_performance = round_all_values(average_test_split_performance)

    return best_results, average_model_performance, average_test_split_performance

def save_summary_to_csv(summary_df, output_path):
    summary_df.to_csv(output_path, index=False)

file_format = 'cluster_results\\classification_summary_{}.pkl'

base = ['ratio', 'MEA', 'count', 'ratiocount']
add_ons = ['', '_single', '_split', '_split_single']
exta_on = ['_smote_single', '_smote']
file_additions = [b+a for b in base for a in add_ons] + [b+e for b in base[2:] for e in exta_on]

file_paths = [file_format.format(addition) for addition in file_additions]

summaries = load_summaries(file_paths)
summary_df = extract_all_results(summaries)

save_summary_to_csv(summary_df, 'cluster_results\\classification_summary.csv')

best_results, avg_model_perf, avg_test_split_perf = create_aggregate_tables(summary_df)

best_results.to_csv('cluster_results\\best_results_summary.csv', index=False)
avg_model_perf.to_csv('cluster_results\\avg_model_performance_summary.csv', index=False)
avg_test_split_perf.to_csv('cluster_results\\avg_test_split_performance_summary.csv', index=False)

for file in file_paths:
    print(file.split('classification_summary_')[1].split('.pkl')[0])
