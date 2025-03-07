import json
import glob

# Define input JSON files (change the pattern as needed)
json_files = glob.glob("./src./*.json")  # Adjust path if necessary

merged_list = []

# Read and merge lists from all JSON files
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        if isinstance(data, list):  # Ensure the file contains a list
            merged_list.extend(data)
        else:
            print(f"Skipping {file}: not a list")

# Write the merged list to an output file
with open("merged.json", "w") as f:
    json.dump(merged_list, f, indent=4)

print("Merged JSON saved as merged.json")
