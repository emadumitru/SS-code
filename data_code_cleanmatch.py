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

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Get all file in directory
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            file_path = os.path.join(root, name)
            new_path = os.path.join(output_path, name)

            filtered_content, actions = filter_file_content(file_path)
            all_actions.update(actions)
            with open(new_path, 'w', encoding='utf-8') as new_file:
                new_file.writelines(filtered_content)

    print("Filtered dataset created successfully.")

    print("All actions found in the dataset:")
    print(all_actions)

if __name__ == "__main__":
    input_dataset_dir = "data_diffs"
    output_dataset_dir = "data_diffs_clean"
    main(input_dataset_dir, output_dataset_dir)
