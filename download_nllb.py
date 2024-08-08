import os
import argparse
import logging
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def load_nllb_dataset_allenai(source_id: str, target_id: str):
    try:
        nllb_dataset = load_dataset("allenai/nllb", f"{source_id}-{target_id}", verification_mode="no_checks", trust_remote_code=True)["train"]
    except ValueError:
        try:
            nllb_dataset = load_dataset("allenai/nllb", f"{target_id}-{source_id}", verification_mode="no_checks", trust_remote_code=True)["train"]
        except ValueError:
            logging.warn(f"Could not download {source_id}-{target_id} pair from allenai/nllb!")
            nllb_dataset = []
    
    return nllb_dataset

def download_nllb(source_id, target_id, output_path, threshold):
    dataset = load_nllb_dataset_allenai(source_id, target_id)
    
    if not dataset:
        print(f"Failed to download dataset for language pair {source_id}-{target_id}")
        return
    
    # Extract source and target sentences
    data = []
    for example in dataset:
        if example["laser_score"] < threshold:
            continue

        source_sentence = example["translation"].get(source_id, "")
        target_sentence = example["translation"].get(target_id, "")
        data.append({
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "laser_score": example["laser_score"],
            "source_sentence_lid": example["source_sentence_lid"],
            "target_sentence_lid": example["target_sentence_lid"],
            "source_sentence_source": example["source_sentence_source"],
            "source_sentence_url": example["source_sentence_url"],
            "target_sentence_source": example["target_sentence_source"],
            "target_sentence_url": example["target_sentence_url"],
        })
    
    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the NLLB dataset and save as a CSV file.")
    parser.add_argument("--source", type=str, required=True, help="Source language ID")
    parser.add_argument("--target", type=str, required=True, help="Target language ID")
    parser.add_argument("--threshold", type=float, required=False, help="LASER3 threshold", default=1.03)
    parser.add_argument("--output", type=str, required=True, help="Output path for the CSV file")
    
    args = parser.parse_args()
    source_id = args.source
    target_id = args.target
    output_path = args.output
    threshold = args.threshold
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    download_nllb(source_id, target_id, output_path, threshold)