from support.github_rethrieve import process_refactorings  
import json

# ghp_PYbx02PfOfLJX  --  dwo2Une4nBEA2bjIk2R9eL4
access_token = ""


json_path = 'dataset analysis\\Silva\\refactorings.json'
output_dir = 'data_code\\code_files_silva'
failed_ids, access_errors, successful_ids = process_refactorings(json_path, output_dir, access_token)

with open('data_code/failed_ids_silva.json', 'w') as json_file:
    json.dump(failed_ids, json_file)
with open('data_code/access_errors_silva.json', 'w') as json_file:
    json.dump(access_errors, json_file)
with open('data_code/successful_ids_silva.json', 'w') as json_file:
    json.dump(successful_ids, json_file)


json_path = 'dataset analysis\\Iman\\Final_Results.json'
output_dir = 'data_code\\code_files_iman'
failed_ids, access_errors, successful_ids = process_refactorings(json_path, output_dir, access_token)

with open('data_code/failed_ids_iman.json', 'w') as json_file:
    json.dump(failed_ids, json_file)
with open('data_code/access_errors_iman.json', 'w') as json_file:
    json.dump(access_errors, json_file)
with open('data_code/successful_ids_iman.json', 'w') as json_file:
    json.dump(successful_ids, json_file)