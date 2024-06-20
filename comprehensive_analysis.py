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

def extract_best_metrics(summaries, file_additions):
    best_metrics = {ref_type: {file: {'best_ac': {}, 'best_tp': {}, 'best_tn': {}} for file in file_additions} for ref_type in set()}
    
    for path, summary in summaries:
        file = path.split('classification_summary_')[1].split('.pkl')[0]
        for ref_type, data in summary.items():
            accuracy, tn, tp, fn, fp = data
            split = ref_type.split('_', 1)
            ref_type, model_params = split[0], split[1]
            if ref_type not in best_metrics:
                best_metrics[ref_type] = {file: {'best_ac': {}, 'best_tp': {}, 'best_tn': {}} for file in file_additions}
            
            if not best_metrics[ref_type][file]['best_ac']:
                best_metrics[ref_type][file]['best_ac'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}
            elif accuracy > best_metrics[ref_type][file]['best_ac']['accuracy']:
                best_metrics[ref_type][file]['best_ac'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}

            if not best_metrics[ref_type][file]['best_tp']:
                best_metrics[ref_type][file]['best_tp'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}
            elif tp > best_metrics[ref_type][file]['best_tp']['tp']:
                best_metrics[ref_type][file]['best_tp'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}

            if not best_metrics[ref_type][file]['best_tn']:
                best_metrics[ref_type][file]['best_tn'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}
            elif tn > best_metrics[ref_type][file]['best_tn']['tn']:
                best_metrics[ref_type][file]['best_tn'] = {'file': file, 'model': model_params, 'accuracy': accuracy, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}
    
    # Filling in missing values with zeros
    for ref_type in best_metrics:
        for file in file_additions:
            for metric in ['best_ac', 'best_tp', 'best_tn']:
                if not best_metrics[ref_type][file][metric]:
                    best_metrics[ref_type][file][metric] = {'file': file, 'model': '', 'accuracy': 0, 'tn': 0, 'tp': 0, 'fn': 0, 'fp': 0}
    
    return best_metrics

def create_summary_dataframe(best_metrics):
    summary_data = []
    for ref_type, file_metrics in best_metrics.items():
        for file, metrics in file_metrics.items():
            summary_data.append([ref_type, 'best_ac', metrics['best_ac']['file'], metrics['best_ac']['model'], metrics['best_ac']['accuracy'], metrics['best_ac']['tn'], metrics['best_ac']['tp'], metrics['best_ac']['fn'], metrics['best_ac']['fp']])
            summary_data.append([ref_type, 'best_tp', metrics['best_tp']['file'], metrics['best_tp']['model'], metrics['best_tp']['accuracy'], metrics['best_tp']['tn'], metrics['best_tp']['tp'], metrics['best_tp']['fn'], metrics['best_tp']['fp']])
            summary_data.append([ref_type, 'best_tn', metrics['best_tn']['file'], metrics['best_tn']['model'], metrics['best_tn']['accuracy'], metrics['best_tn']['tn'], metrics['best_tn']['tp'], metrics['best_tn']['fn'], metrics['best_tn']['fp']])
    
    summary_df = pd.DataFrame(summary_data, columns=['Refactoring Type', 'Metric', 'Source File', 'Model and Params', 'Accuracy', 'TN', 'TP', 'FN', 'FP'])
    return summary_df

def save_summary_to_csv(summary_df, output_path):
    summary_df.to_csv(output_path, index=False)

file_format = 'cluster_results\\classification_summary_{}.pkl'

base = ['ratio', 'MEA', 'count', 'ratiocount']
add_ons = ['', '_single', '_split', '_split_single']
exta_on = ['_smote_single', '_smote']
file_additions = [b+a for b in base for a in add_ons] + [b+e for b in base[2:] for e in exta_on]

file_paths = [file_format.format(addition) for addition in file_additions]

summaries = load_summaries(file_paths)
best_metrics = extract_best_metrics(summaries, file_additions)
summary_df = create_summary_dataframe(best_metrics)
save_summary_to_csv(summary_df, 'cluster_results\\classification_summary.csv')

for file in file_paths:
    print(file.split('classification_summary_')[1].split('.pkl')[0]) # Print file names