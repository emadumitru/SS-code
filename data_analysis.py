import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

# Load the JSON data pre and post
with open('data_code\\successful_ids_silva.json', 'r') as file:
    data_post = json.load(file)
with open('dataset analysis\\Silva\\refactorings.json', 'r') as file:
    full_data = json.load(file)

data_pre = {}
for ref in full_data:
    ref_id = str(ref['id'])
    data_pre[ref_id] = [refactoring['type'] for refactoring in ref['refactorings']]

# Enhanced function to analyze the refactoring data
def analyze_data(data):
    details_by_id = defaultdict(Counter)
    single_refactoring = Counter()
    multiple_refactoring = Counter()
    total_refactoring_count = Counter()
    
    for ref_id, refactorings in data.items():
        types = set(refactorings)
        details_by_id[ref_id].update(refactorings)
        if len(types) == 1:
            single_refactoring[list(types)[0]] += 1
        else:
            for ref_type in types:
                multiple_refactoring[ref_type] += 1
        for ref_type in types:
            total_refactoring_count[ref_type] += 1
            
    return single_refactoring, multiple_refactoring, total_refactoring_count, details_by_id

# Analyze both datasets
orig_single, orig_multiple, orig_total, details_pre = analyze_data(data_pre)
retr_single, retr_multiple, retr_total, details_post = analyze_data(data_post)

# Prepare data for visualization
single_types_combined = pd.DataFrame({'Original Single': orig_single, 'Retrieved Single': retr_single}).fillna(0)
multiple_types_combined = pd.DataFrame({'Original Mixed': orig_multiple, 'Retrieved Mixed': retr_multiple}).fillna(0)
total_types_combined = pd.DataFrame({'Original Total': orig_total, 'Retrieved Total': retr_total}).fillna(0)

# Combine all data into one DataFrame for CSV export
combined_data = pd.concat([total_types_combined, multiple_types_combined, single_types_combined], axis=1)

# Save the combined data to CSV
combined_data.to_csv('analysis_results\\combined_refactoring_analysis.csv')

# Create a directory for the results if it doesn't exist
output_dir = 'analysis_results\\'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Visualizations
def add_labels(ax, show_values=True):
    if show_values:
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center")

def plot_data(df, title, filename):
    ax = df.plot(kind='bar', figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    add_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))

# Plot and save data
plot_data(single_types_combined, 'Count of Files with Single Refactoring Type', 'single_refactoring_types.png')
plot_data(multiple_types_combined, 'Count of Files with Mixed Refactoring Types', 'mixed_refactoring_types.png')
plot_data(total_types_combined, 'Total Refactoring Types Count', 'total_refactoring_types.png')

# Save textual summary
summary_path = os.path.join(output_dir, 'analysis_summary.txt')
with open(summary_path, 'w') as file:
    file.write("Analysis Summary of Refactoring Types\n")
    file.write("Single Refactoring Types:\n")
    file.write(single_types_combined.to_string() + "\n")
    file.write("Mixed Refactoring Types:\n")
    file.write(multiple_types_combined.to_string() + "\n")
    file.write("Total Refactoring Types:\n")
    file.write(total_types_combined.to_string() + "\n")

# Output paths to confirm files were saved
print(os.listdir(output_dir))
