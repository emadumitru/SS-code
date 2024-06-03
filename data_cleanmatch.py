import os

def filter_file_content(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Split into blocks that start with line sthata are "==="
    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == '===':
            if current_block != []:
                blocks.append(current_block)
                current_block = []
        current_block.append(line)

    cleaned_blocks = [block for block in blocks if len(block) > 1 and block[1].strip() != 'match']

    # Reconstruct the content
    filtered_content = []
    for block in cleaned_blocks:
        filtered_content.extend(block)

    # Remaining actions
    actions = set([block[1].strip() for block in cleaned_blocks])
    
    return filtered_content, actions


def main(dataset_path, output_path):
    # set all actions
    all_actions = set()
    # Create the output directory structure mirroring the input structure
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            file_path = os.path.join(root, name)
            # Construct the corresponding output file path
            relative_path = os.path.relpath(file_path, dataset_path)
            new_path = os.path.join(output_path, relative_path)
            new_dir = os.path.dirname(new_path)

            if not os.path.exists(new_dir):
                os.makedirs(new_dir, exist_ok=True)

            # Filter and write the content
            filtered_content, actions = filter_file_content(file_path)
            all_actions.update(actions)
            with open(new_path, 'w', encoding='utf-8') as new_file:
                new_file.writelines(filtered_content)

    # Delete refactoring with less than 2 file sin _single
    for root, dirs, files in os.walk(output_path):
        for name in dirs:
            if name.endswith("_single"):
                dir_path = os.path.join(root, name)
                if len(os.listdir(dir_path)) < 2:
                    print(f"Deleting {dir_path}")
                    # delete all in _single
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        os.remove(file_path)
                    os.rmdir(dir_path)
                    mixed_dir = os.path.join(root, name.replace("_single", "_mixed"))
                    if os.path.exists(mixed_dir):
                        print(f"Deleting {mixed_dir}")
                        # delete all in _mixed
                        for file in os.listdir(mixed_dir):
                            file_path = os.path.join(mixed_dir, file)
                            os.remove(file_path)
                        os.rmdir(mixed_dir)

    print("Filtered dataset created successfully.")

    print("All actions found in the dataset:")
    print(all_actions)

if __name__ == "__main__":
    input_dataset_dir = "dataset"
    output_dataset_dir = "dataset_clean"
    main(input_dataset_dir, output_dataset_dir)
