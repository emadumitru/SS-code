import os
import subprocess

def run_gumtree_and_save_output(directory, output_dir, gumtree=r'C:\\Program Files\\GumTree\\gumtree-3.0.0\\bin'):
    files = os.listdir(directory)
    pairs = {}
    
    # Update PATH in the script
    os.environ['PATH'] += os.pathsep + gumtree
    # Organize files into pairs
    for file in files:
        if file.endswith('_pre.java'):
            file_id = file.split('_pre.java')[0]
            if file_id not in pairs:
                pairs[file_id] = [file, '']
            else:
                pairs[file_id][0] = file
        elif file.endswith('_ref.java'):
            file_id = file.split('_ref.java')[0]
            if file_id in pairs:
                pairs[file_id][1] = file
            else:
                pairs[file_id] = ['', file]

    # Check all pairs have both pre and ref files
    to_pop = []
    for pair in pairs:
        if pairs[pair][0] == '' or pairs[pair][1] == '':
            to_pop.append(pair)
    for pair in to_pop:
        pairs.pop(pair)
        print(f'Pair {pair} does not have both pre and ref files')

    # Run GumTree and save diffs to files
    for file_id, pair in pairs.items():
        pre_file = os.path.join(directory, pair[0])
        ref_file = os.path.join(directory, pair[1])
        diff_file_path = os.path.join(output_dir, f'{file_id}_diff.md')
        if os.path.exists(pre_file) and os.path.exists(ref_file):
            cmd = "gumtree textdiff {} {}".format(pre_file, ref_file)
            with open(diff_file_path, 'w') as diff_file:
                subprocess.run(cmd, shell=True, stdout=diff_file, text=True)

def copy_files_as_java(input_dir, output_dir):
    # test if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('.md'):
            file_name = file.split('.md')[0]
            try:
                with open(os.path.join(input_dir, file), 'r') as md_file:
                    lines = md_file.readlines()
                with open(os.path.join(output_dir, f'{file_name}.java'), 'w') as java_file:
                    java_file.writelines(lines)
            except:
                print(f'Error in file {file_name}')


def run_diffing(input_dir, output_dir, middle_dir="temporary/", delete_middle=True):
    # Copy files as java
    copy_files_as_java(input_dir, middle_dir)
    # Run GumTree
    run_gumtree_and_save_output(middle_dir, output_dir)
    if delete_middle:
        # Delete temporary files
        files = os.listdir(middle_dir)
        for file in files:
            os.remove(os.path.join(middle_dir, file))
        os.rmdir(middle_dir)
