import os
import shutil
import json

# Assuming the JSON data that contains mapping of refactoring types and associated diff files is available
with open('data_code\\successful_ids_silva.json', 'r') as file:
    data_post = json.load(file)

# Path to the diffs
diffs_path = "data_diffs\\"  

# Create the dataset directory if it does not exist
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Create subdirectories for each refactoring type
refactoring_types = set()
for _, refactorings in data_post.items():
    for ref_type in refactorings:
        refactoring_types.add(ref_type)

# Dictionary to keep track of files for each type
ref_type_files = {ref_type: {'single': [], 'mixed': []} for ref_type in refactoring_types}

# Assign files to the appropriate type and category
for file_id, ref_types in data_post.items():
    unique_ref_types = set(ref_types)
    if len(unique_ref_types) == 1:
        ref_type_files[list(unique_ref_types)[0]]['single'].append(file_id)
    else:
        for ref_type in unique_ref_types:
            ref_type_files[ref_type]['mixed'].append(file_id)

# Create directories and copy files
for ref_type, categories in ref_type_files.items():
    for category, files in categories.items():
        category_dir = os.path.join(dataset_dir, f"{ref_type}_{category}")
        os.makedirs(category_dir, exist_ok=True)
        for file_name in files:
            src = os.path.join(diffs_path, f"{file_name}_diff.md") 
            dst = os.path.join(category_dir, f"{file_name}_diff.md")
            if os.path.exists(src):
                shutil.copy2(src, dst)

print("Directories created and files organized successfully.")