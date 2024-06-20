import pandas as pd

# Load the CSV data
csv_file = 'cluster_results\\best_results_summary.csv'
df = pd.read_csv(csv_file)

# Initialize a dictionary to store the best accuracy values for each refactoring type
best_accuracy_data = {}

# Iterate over the DataFrame to find the best accuracy for each refactoring type
for index, row in df.iterrows():
    ref_type = row['Refactoring Type']
    source_file = row['Source File']
    source_file = str(source_file).replace('_', ' ')
    metric = row['metric']
    accuracy = row['Accuracy']
    tn = row['TN']
    tp = row['TP']
    model = row['Model']
    
    if metric == 'Accuracy':
        if ref_type not in best_accuracy_data or accuracy > best_accuracy_data[ref_type]['BA']:
            best_accuracy_data[ref_type] = {
                'Source File': source_file,
                'BA': accuracy,
                'TN': tn,
                'TP': tp,
                'Model': model
            }

# Generate LaTeX table content
table_content = """\\begin{table*}
    \\begin{tabular}{c|c|c|c|c}
    \\toprule
    Refactoring Type & Approach & BA (\%) & TN (\%) & TP (\%) \\\\
    \\midrule
"""
models = []

for ref_type, values in best_accuracy_data.items():
    table_content += f"    {ref_type} & {values['Source File']} & {round(values['BA']*100)} & {values['TN']} & {values['TP']} \\\\\n"
    models.append((ref_type, values['Model']))

table_content += """    \\bottomrule
    \\end{tabular}
    \\caption{Table of best balanced accuracy (BA), true negative (TN), and true positive (TP) percentages for each refactoring type}
    \\label{tab:refactoring_best_accuracy}
\\end{table*}
"""

# Save the table content to a text file
output_file = 'output\\overleaf_max_table.txt'
with open(output_file, 'w') as f:
    f.write(table_content)

print("LaTeX table content has been written to", output_file)
print("Models used:", models)
