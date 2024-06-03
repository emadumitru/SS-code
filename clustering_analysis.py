import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

path = 'analysis_results\\'


def add_labels(ax):
    """Add labels above each bar in the axes, displaying its height with 3 decimals."""
    for rect in ax.patches:
        height = rect.get_height()
        if height < 1:
            ax.annotate('{}'.format(round(height * 100, 1)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        else:
            ax.annotate('{}'.format(round(height, 3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
def plot_data(df, title, filename):
    ax = df.plot(kind='bar', figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    add_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))
    plt.close() 

# Load the results
with open(path + 'classification_summary.pkl', 'rb') as f:
    summary = pickle.load(f)

# Data for plots
refactoring_types = list(summary.keys())
accuracies = [summary[key][0] for key in refactoring_types]
TP = [summary[key][4] for key in refactoring_types]
TN = [summary[key][1] for key in refactoring_types]
FP = [summary[key][2] for key in refactoring_types]
FN = [summary[key][3] for key in refactoring_types]

# Create DataFrame for accuracies and plot
accuracy_df = pd.DataFrame({'Accuracy': accuracies}, index=refactoring_types)
plot_data(accuracy_df, 'Accuracy for Each Refactoring Type', 'accuracy_refactoring_types.png')

# Create DataFrame for TP, TN, FP, FN and plot
performance_df = pd.DataFrame({'True Positives': TP, 'True Negatives': TN, 'False Positives': FP, 'False Negatives': FN}, index=refactoring_types)
plot_data(performance_df, 'Number of TP, TN, FP, FN by Refactoring Type', 'performance_refactoring_types.png')