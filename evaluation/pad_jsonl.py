import json
import os

def create_empty_json(idx, template):
    empty_json = {}
    for key in template:
        if key == 'idx':
            empty_json[key] = idx
        else:
            value = template[key]
            if isinstance(value, dict):
                empty_json[key] = create_empty_structure(value)
            elif isinstance(value, list):
                empty_json[key] = []
                if len(value) > 0:
                    # Assuming list of dicts
                    first_elem = value[0]
                    empty_elem = create_empty_structure(first_elem)
                    num_elems = len(value)
                    empty_json[key] = [empty_elem for _ in range(num_elems)]
                else:
                    empty_json[key] = []
            else:
                empty_json[key] = ''
    return empty_json

def create_empty_structure(struct):
    if isinstance(struct, dict):
        return {k: '' if not isinstance(v, list) else [] for k, v in struct.items()}
    elif isinstance(struct, list):
        if len(struct) > 0:
            return [create_empty_structure(struct[0])]
        else:
            return []
    else:
        return ''

# Get the JSONL file path from the environment variable
jsonl_file = os.getenv('EVALUATION_FILE_PATH')

if not jsonl_file:
    print("Error: The environment variable EVALUATION_FILE_PATH is not set.")
    exit(1)

# Prepare the new filename
file_root, file_ext = os.path.splitext(jsonl_file)
new_jsonl_file = f"{file_root}_padded{file_ext}"

# Read existing JSONL file to get the template and last idx
try:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Error: File not found at {jsonl_file}.")
    exit(1)

if not lines:
    print("The file is empty.")
    exit()

# Parse all existing entries
entries = [json.loads(line) for line in lines]

template = entries[0]
last_idx = entries[-1].get('idx', 0)

# Write existing entries to the new file
with open(new_jsonl_file, 'w', encoding='utf-8') as f:
    for entry in entries:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')

    # Append new entries with empty values
    for idx in range(last_idx + 1, 1001):
        empty_entry = create_empty_json(idx, template)
        json_line = json.dumps(empty_entry)
        f.write(json_line + '\n')

print(f"New padded JSONL file created at: {new_jsonl_file}")
