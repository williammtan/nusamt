import gzip
import pandas as pd

def read_gz_file(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        lines = set(f.read().splitlines())
    return lines

def count_matches(source_lines, target_file):
    target_lines = read_gz_file(target_file)
    matches = source_lines.intersection(target_lines)
    return len(matches)

def main(source_file, target_files):
    source_lines = read_gz_file(source_file)
    results = []

    for target_file in target_files:
        match_count = count_matches(source_lines, target_file)
        results.append((target_file, match_count))

    df = pd.DataFrame(results, columns=['Target File', 'Match Count'])
    print(df)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count matches in target files based on source file.")
    parser.add_argument("source_file", type=str, help="Path to the source .gz file")
    parser.add_argument("target_files", type=str, nargs='+', help="Paths to the target .gz files")

    args = parser.parse_args()
    main(args.source_file, args.target_files)