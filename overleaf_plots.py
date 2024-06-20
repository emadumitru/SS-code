import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('cluster_results\\best_results_summary.csv')
save_folder_path = 'output\\'

# Process the data
def preprocess_data(data):
    data = data[data['metric'] == 'Accuracy']
    return data

processed_data = preprocess_data(data)

# Define a function to plot the data based on file type and save the plot
def plot_data_by_file_type(data, refactoring_type, save_path):
    # Filter data for the specific refactoring type
    filtered_data = data[data['Refactoring Type'] == refactoring_type]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x_labels = []
    
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        approach = row['Approach']
        if approach == 'ratiocount':
            approach = 'count+'
        if approach == 'MEA':
            approach = 'n-gram'
        type_split = row['Type Split']
        if type_split:
            type_split = 'split'
        else:
            type_split = ''
        
        if 'smote' in row['Source File']:
            color = 'pink'
            offset = 0.3
        else:
            type_file = row['Type File']
            color = 'orange' if type_file else 'blue'
            offset = 0.1 if color == 'orange' else -0.1
        
        tp = row['TP']
        tn = row['TN']
        ba = (row['Accuracy']*100)
        
        # Plot TP, TN, and Balanced Accuracy (BA) with slight offset for colors
        label_pos = f"{approach} {type_split}"
        
        if label_pos not in x_labels:
            x_labels.append(label_pos)
        
        pos = x_labels.index(label_pos)
        
        ax.plot(pos + offset, tp, '+', markersize=10, color=color)
        ax.plot(pos + offset, tn, '_', markersize=10, color=color)
        ax.plot(pos + offset, ba, 'o', markersize=10, color=color)
        ax.vlines(pos + offset, tp, ba, colors=color, linestyles='dashed', alpha=0.2)
        ax.vlines(pos + offset, tn, ba, colors=color, linestyles='dashed', alpha=0.2)
    
    # Set x-ticks
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Approach')
    ax.set_ylabel('Values')
    ax.set_title(f'{refactoring_type}', fontsize=15)
    plt.tight_layout()

    # Alternating background colors
    for i in range(len(x_labels)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='lightgrey', alpha=0.2)
        else:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='white', alpha=0.5)

    # Remove grid lines
    ax.grid(False)

    #transparent background
    fig.patch.set_alpha(0)
    
    # Save the plot
    file_name = f"file_type_{refactoring_type}.png"
    plt.savefig(os.path.join(save_path, file_name))
    
    plt.close()

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_data_by_split_type(data, refactoring_type, save_path):
    # Filter data for the specific refactoring type
    filtered_data = data[data['Refactoring Type'] == refactoring_type]

    n_approach = filtered_data['Approach'].nunique()
    n_type = filtered_data['Type File'].nunique()
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(n_approach * n_type, 5))
    
    x_labels = []
    
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        approach = row['Approach']
        if approach == 'ratiocount':
            approach = 'count+'
        if approach == 'MEA':
            approach = 'n-gram'
        type_split = row['Type Split']
        
        if 'smote' in row['Source File']:
            color = 'pink'
            offset = 0.3
        else:
            color = 'orange' if type_split else 'blue'
            offset = 0.1 if color == 'orange' else -0.1
        
        type_file = row['Type File']
        if type_file:
            type_file = 'single'
        else:
            type_file = 'mixed'
        tp = row['TP']
        tn = row['TN']
        ba = (row['Accuracy'] * 100)
        
        # Plot TP, TN, and Balanced Accuracy (BA) with slight offset for colors
        label_pos = f"{approach} {type_file}"
        
        if label_pos not in x_labels:
            x_labels.append(label_pos)
        
        pos = x_labels.index(label_pos)
        
        ax.plot(pos + offset, tp, '+', markersize=10, color=color)
        ax.plot(pos + offset, tn, '_', markersize=10, color=color)
        ax.plot(pos + offset, ba, 'o', markersize=10, color=color)
        ax.vlines(pos + offset, tp, ba, colors=color, linestyles='dashed', alpha=0.2)
        ax.vlines(pos + offset, tn, ba, colors=color, linestyles='dashed', alpha=0.2)
    
    # Set x-ticks
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
    
    # Customize the plot
    ax.set_xlabel('Approach')
    ax.set_ylabel('Values')
    ax.set_title(f'{refactoring_type}', fontsize=15)
    plt.tight_layout()
    
    # Alternating background colors
    for i in range(len(x_labels)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='lightgrey', alpha=0.2)
        else:
            ax.axvspan(i - 0.5, i + 0.5, facecolor='white', alpha=0.5)

    # Remove grid lines
    ax.grid(False)

    #transparent background
    fig.patch.set_alpha(0)
    
    # Save the plot
    file_name = f"split_type_{refactoring_type}.png"
    plt.savefig(os.path.join(save_path, file_name))
    
    plt.close()


# Ensure the folder exists
os.makedirs(save_folder_path, exist_ok=True)

# Plot data for each refactoring type and save the plots
refactoring_types = processed_data['Refactoring Type'].unique()
for refactoring_type in refactoring_types:
    plot_data_by_file_type(processed_data, refactoring_type, save_folder_path)
    plot_data_by_split_type(processed_data, refactoring_type, save_folder_path)
