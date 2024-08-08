import gzip
import json
import os
import argparse
import csv
from itertools import zip_longest

def process_files(file_paths, output_file_path, keys=None):
    if keys and len(keys) != len(file_paths):
        raise ValueError("The number of keys must match the number of files")
    
    file_handles = []
    is_jsonl = []

    for file_path in file_paths:
        if file_path.endswith('.gz'):
            file_handles.append(gzip.open(file_path, 'rt'))
            is_jsonl.append(False)
        elif file_path.endswith('.jsonl'):
            file_handles.append(open(file_path, 'r'))
            is_jsonl.append(True)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

    if not keys:
        keys = [os.path.basename(file_path) for file_path in file_paths]
    
    result = []

    for lines in zip_longest(*file_handles, fillvalue=None):
        line_dict = {}
        for idx, (key, line) in enumerate(zip(keys, lines)):
            if line is not None:
                line = line.strip()
                if is_jsonl[idx]:
                    try:
                        line = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON in file {file_paths[idx]}: {line}")
                line_dict[key] = line
        result.append(line_dict)
    
    for file_handle in file_handles:
        file_handle.close()
    
    return result

def output_as_json(data, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def output_as_csv(data, output_file_path, keys):
    with open(output_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Process .gz and .jsonl files and output to a JSON or CSV file.')
    parser.add_argument('files', metavar='FILE', type=str, nargs='+', help='List of .gz or .jsonl files to process')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to the output file')
    parser.add_argument('-k', '--keys', type=str, nargs='*', help='Optional list of keys to use instead of filenames')
    parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json', help='Output format: json or csv (default: json)')

    args = parser.parse_args()

    data = process_files(args.files, args.output, args.keys)
    
    if args.format == 'json':
        output_as_json(data, args.output)
    else:
        keys = args.keys if args.keys else [os.path.basename(file_path) for file_path in args.files]
        output_as_csv(data, args.output, keys)

if __name__ == "__main__":
    main()