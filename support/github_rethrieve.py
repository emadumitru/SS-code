from urllib.parse import urlparse
import requests
import base64
import json
import os


def process_refactorings(json_path, output_dir, access_token, show_progress=False):
    def get_commit_details(repo_url, commit_sha):
        api_url = f"https://api.github.com/repos/{repo_url}/commits/{commit_sha}"
        headers = {'Authorization': f'token {access_token}'}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            if show_progress:
                print(f"Failed to fetch commit details. Status code: {response.status_code}, Message: {response.reason}")
            return None

    def get_file_content(repo_url, commit_sha, file_path):
        api_url = f"https://api.github.com/repos/{repo_url}/contents/{file_path}?ref={commit_sha}"
        headers = {'Authorization': f'token {access_token}'}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            file_data = response.json()
            return file_data.get('content', None)
        else:
            return None

    def extract_commit_code(commit_details, output_file_path, repo_url, is_pre_commit):
        if is_pre_commit:
            commit_sha = commit_details['parents'][0]['sha']  # use parent commit to extract pre-commit source code
        else:
            commit_sha = commit_details['sha']  # use current commit to extract post-commit source code

        current_commit_files = commit_details['files']

        with open(output_file_path, 'ab') as output_file:
            for file_change in current_commit_files:
                file_path = file_change['filename']
                if not file_path.endswith('.java'):
                    continue  # keeping only Java files

                file_content_base64 = get_file_content(repo_url, commit_sha, file_path)

                if file_content_base64 is not None:
                    file_content = base64.b64decode(file_content_base64).decode('utf-8', errors='replace')

                    output_file.write(f"\n\nFile: {file_path}\n".encode('utf-8'))
                    output_file.write(file_content.encode('utf-8'))

    with open(json_path) as json_file:
        refactorings_data = json.load(json_file)

    failed_ids = []
    access_errors = []
    successful_ids = {}

    itter = 0

    for entry in refactorings_data:
        repo_url = entry['repository']
        try:
            name = entry['id']
        except:
            name = itter
            itter += 1
        parsed_url = urlparse(repo_url)
        owner, repo = parsed_url.path.lstrip('/').split('/', 1)
        repo = repo.rstrip('.git')
        repo_url = f"{owner}/{repo}"
        commit_sha = entry['sha1']
        output_file_path_pre = os.path.join(output_dir, f"{name}_pre.md")
        output_file_path_post = os.path.join(output_dir, f"{name}_ref.md")

        # Create directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Fetch commit details
        commit_details = get_commit_details(repo_url, commit_sha)
        if commit_details is not None:  
            try:
                # Extract pre-commit code
                with open(output_file_path_pre, 'wb') as output_file:
                    extract_commit_code(commit_details, output_file_path_pre, repo_url, is_pre_commit=True)

                # Extract post-commit code
                with open(output_file_path_post, 'wb') as output_file:
                    extract_commit_code(commit_details, output_file_path_post, repo_url, is_pre_commit=False)
                
                # If successful, record refactoring types
                successful_ids[name] = [ref['type'] for ref in entry.get('refactorings', [])]
            except Exception as e:
                failed_ids.append(name)
        else:
            access_errors.append(name)

    return failed_ids, access_errors, successful_ids