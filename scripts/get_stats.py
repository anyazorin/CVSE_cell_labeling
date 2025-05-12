import json
from collections import defaultdict

# Load your JSON file (replace 'data.json' with your filename)
with open('annotations/output.json', 'r') as f:
    entries = json.load(f)
print(len(entries))
print(entries[0])
# Count successes and failures
num_success = sum(1 for e in entries if e.get('success') is True)
num_failure = sum(1 for e in entries if e.get('success') is False)

# Report
print(f"Number of successes: {num_success}")
print(f"Number of failures: {num_failure}")

# group_success = defaultdict(lambda: False)

# # For each record, if it’s a success, mark that group True
# for e in entries:
#     key = (e['img_name'], e['cell_num'])
#     if e.get('success', False):
#         group_success[key] = True

# # Now count groups
# num_success_groups = sum(1 for success in group_success.values() if success)
# num_failure_groups = sum(1 for success in group_success.values() if not success)
# total_groups       = len(group_success)

# print(f"Unique success groups: {num_success_groups}")
# print(f"Unique failure groups: {num_failure_groups}")
# print(f"Total unique groups:   {total_groups}")

group_status = {}

for e in entries:
    key = (e['img_name'], e['cell_num'])
    was_success = bool(e.get('success', False))
    if key in group_status:
        # once True, stays True
        group_status[key] = group_status[key] or was_success
    else:
        # first time we see this group
        group_status[key] = was_success

# Now count
num_success_groups = sum(1 for ok in group_status.values() if ok)
num_failure_groups = sum(1 for ok in group_status.values() if not ok)
total_groups       = len(group_status)

print(f"Unique success groups: {num_success_groups}")
print(f"Unique failure groups: {num_failure_groups}")
print(f"Total unique groups:   {total_groups}")