import os

def replace_substring_in_file(file_path, old_substring, new_substring):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = file.read()
        
        new_data = file_data.replace(old_substring, new_substring)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_data)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def recursively_replace_substring(directory, old_substring, new_substring):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            replace_substring_in_file(file_path, old_substring, new_substring)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    old_substring = input("Enter the substring to replace: ")
    new_substring = input("Enter the new substring: ")

    recursively_replace_substring(directory, old_substring, new_substring)

    print("Replacement complete.")