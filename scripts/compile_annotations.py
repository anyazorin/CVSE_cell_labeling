import json
import os
import argparse


def process_annotations(parent_dir):
    """Scan all cell folders in parent_dir and write combined_nucleus_annotation.json."""
    combined_nucleus_entries = []

    if not os.path.isdir(parent_dir):
        print(f" Skipping {parent_dir!r}: not a directory")
        return

    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)

        # Only process directories named 'cell<number>'
        if os.path.isdir(folder_path) and folder_name.startswith('cell'):
            try:
                cell_num = int(folder_name.replace('cell', ''))
            except ValueError:
                continue

            json_filename = f"{folder_name}_annotation.json"
            json_path = os.path.join(folder_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Keep only entries with nucleus == True
                for entry in data:
                    if entry.get("nucleus") is True:
                        entry["cell_num"] = cell_num
                        combined_nucleus_entries.append(entry)

    # write out
    output_path = os.path.join(parent_dir, 'combined_nucleus_annotation.json')
    with open(output_path, 'w') as f:
        json.dump(combined_nucleus_entries, f, indent=4)

    print(f"Combined {len(combined_nucleus_entries)} entries into {output_path}")


def process_outputs(parent_dir= "annotations"):
    """Scan all cell folders in parent_dir and write combined_nucleus_annotation.json."""
    output = []

    if not os.path.isdir(parent_dir):
        print(f" Skipping {parent_dir!r}: not a directory")
        return

    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        json_filename = "output.json"
        json_path = os.path.join(folder_path, json_filename)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Keep only entries with nucleus == True
            for entry in data:
                output.append(entry)

    # write out
    output_path = os.path.join(parent_dir, 'output.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Combined {len(output)} entries into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compile annotations for image(s).")
    parser.add_argument(
        '-i', '--imgname',
        help='Name of the image file (without path). If omitted, all folders under annotations/ will be processed.'
    )
    args = parser.parse_args()

    base_dir = 'annotations'

    if args.imgname:
        # single-image mode
        dirs = [f"{args.imgname}_annotations"]
    else:
        # batch mode: every *_annotations folder
        dirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_annotations')
        ]

    if not dirs:
        print("No annotation folders found to process.")
        return

    for d in dirs:
        parent_dir = os.path.join(base_dir, d)
        print(f"\n Processing {parent_dir} …")
        process_annotations(parent_dir)


if __name__ == '__main__':
    # main()
    process_outputs()
# parser = argparse.ArgumentParser(description="Compile annotations for image.")
# parser.add_argument('-i', '--imgname', required=True, help='Name of the image file (without path)')

# args = parser.parse_args()
# imgname = args.imgname

# # Set the parent directory path where the 'cell0', 'cell1', etc. folders are
# parent_dir = f'annotations/{imgname}_annotations'

# # List to hold all the nucleus=true entries
# combined_nucleus_entries = []

# # Go through each subfolder
# for folder_name in os.listdir(parent_dir):
#     folder_path = os.path.join(parent_dir, folder_name)
    
#     # Check if it's a folder and matches 'cell' pattern
#     if os.path.isdir(folder_path) and folder_name.startswith('cell'):
#         # Extract the cell number
#         try:
#             cell_num = int(folder_name.replace('cell', ''))
#         except ValueError:
#             continue  # Skip folders that don't match 'cell[number]'

#         json_filename = f"{folder_name}_annotation.json"
#         json_path = os.path.join(folder_path, json_filename)

#         if os.path.exists(json_path):
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
#                 # Filter entries where "nucleus" == True
#                 nucleus_entries = [entry for entry in data if entry.get("nucleus") is True]
#                 # Add cell_num to each entry
#                 for entry in nucleus_entries:
#                     entry["cell_num"] = cell_num
#                 combined_nucleus_entries.extend(nucleus_entries)

# # Write the combined entries to a new JSON file
# output_path = os.path.join(parent_dir, 'combined_nucleus_annotation.json')
# with open(output_path, 'w') as f:
#     json.dump(combined_nucleus_entries, f, indent=4)

# print(f"Combined {len(combined_nucleus_entries)} entries into {output_path}")
