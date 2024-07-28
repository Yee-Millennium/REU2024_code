import os

def replace_spaces_with_commas(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    modified_content = content.replace(' ', ',')
    with open(file_path, 'w') as file:
        file.write(modified_content)

directory = 'data'  

for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Process only text files
        file_path = os.path.join(directory, filename)
        replace_spaces_with_commas(file_path)
        print(f"Modified {file_path}")
