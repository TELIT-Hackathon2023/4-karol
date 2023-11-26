import json

# Load the 25MB JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Write the reformatted JSON data with an indentation of 4 spaces
with open('data_indented.json', 'w') as file:
    json.dump(data, file, indent=4)
