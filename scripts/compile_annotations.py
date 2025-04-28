import json
import os
import argparse

parser = argparse.ArgumentParser(description="Compile annotations for image.")
parser.add_argument('-i', '--imgname', required=True, help='Name of the image file (without path)')

args = parser.parse_args()
imgname = args.imgname

# Set the parent directory path where the 'cell0', 'cell1', etc. folders are
parent_dir = f'annotations/{imgname}_annotations'

# List to hold all the nucleus=true entries
combined_nucleus_entries = []

# Go through each subfolder
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    
    # Check if it's a folder and matches 'cell' pattern
    if os.path.isdir(folder_path) and folder_name.startswith('cell'):
        # Extract the cell number
        try:
            cell_num = int(folder_name.replace('cell', ''))
        except ValueError:
            continue  # Skip folders that don't match 'cell[number]'

        json_filename = f"{folder_name}_annotation.json"
        json_path = os.path.join(folder_path, json_filename)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Filter entries where "nucleus" == True
                nucleus_entries = [entry for entry in data if entry.get("nucleus") is True]
                # Add cell_num to each entry
                for entry in nucleus_entries:
                    entry["cell_num"] = cell_num
                combined_nucleus_entries.extend(nucleus_entries)

# Write the combined entries to a new JSON file
output_path = os.path.join(parent_dir, 'combined_nucleus_annotation.json')
with open(output_path, 'w') as f:
    json.dump(combined_nucleus_entries, f, indent=4)

print(f"Combined {len(combined_nucleus_entries)} entries into {output_path}")
