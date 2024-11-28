import os
import json
import argparse
import pandas as pd
from itertools import permutations
import requests
import zipfile

# Define constants
FLORES_URL = "https://github.com/openlanguagedata/flores/releases/download/v2.0-rc.3/floresp-v2.0-rc.3.zip"
SUBDIRS = ['dev', 'devtest']
OPUS_LANGUAGE_IDS = ['ban', 'ace', 'bjn', 'bug', 'min', 'su', 'jv', 'id', 'en']
LANG_IDS = ['ban_Latn', 'ace_Latn', 'bjn_Latn', 'bug_Latn', 'min_Latn', 'sun_Latn', 'jav_Latn', 'ind_Latn', 'eng_Latn']
ZIP_PASSWORD = b"multilingual machine translation"  # Convert password to bytes


def readlines(filepath):
    with open(filepath) as f:
        return [l for l in f.read().split('\n') if l != ""]

def download_and_extract_flores(download_url, extract_to, password):
    zip_path = os.path.join(extract_to, "floresp-v2.0-rc.3.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Flores dataset from {download_url}...")
        response = requests.get(download_url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)
        print("Download complete.")

    if not os.path.exists(os.path.join(extract_to, "floresp-v2.0-rc.3")):
        print(f"Extracting Flores dataset to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to, pwd=password)
        print("Extraction complete.")

def main(save_dir):
    # Create save_dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Directory to save Flores dataset
    FLORES_DIR = os.path.join(save_dir, "floresp-v2.0-rc.3")

    # Download and extract Flores dataset if not already done
    download_and_extract_flores(FLORES_URL, save_dir, ZIP_PASSWORD)

    for idx1, idx2 in permutations(range(len(OPUS_LANGUAGE_IDS)), 2):
        src, tgt = OPUS_LANGUAGE_IDS[idx1], OPUS_LANGUAGE_IDS[idx2]
        src_nllb, tgt_nllb = LANG_IDS[idx1], LANG_IDS[idx2]
        os.makedirs(os.path.join(save_dir, f"{src}{tgt}"), exist_ok=True)
        
        src_dev = readlines(os.path.join(FLORES_DIR, "dev", f"dev.{src_nllb}"))
        tgt_dev = readlines(os.path.join(FLORES_DIR, "dev", f"dev.{tgt_nllb}"))
        src_devtest = readlines(os.path.join(FLORES_DIR, "devtest", f"devtest.{src_nllb}"))
        tgt_devtest = readlines(os.path.join(FLORES_DIR, "devtest", f"devtest.{tgt_nllb}"))


        # TRAIN: train.src-tgt.json
        with open(os.path.join(save_dir, f"{src}{tgt}", f"train.{src}-{tgt}.json"), 'w') as f:
            translation_dicts = []
            for i in range(len(src_dev)):
                translation_dicts.append({"translation": {
                    src: src_dev[i],
                    tgt: tgt_dev[i]
                }})

            json.dump(translation_dicts, f)


        with open(os.path.join(save_dir, f"{src}{tgt}", f"valid.{src}-{tgt}.json"), 'w') as f:
            json.dump([{
                "translation": {
                    src: src_dev[-1],
                    tgt: tgt_dev[-1]
                }
            }], f)
        

        # TEST: test.src-tgt.json + test.src-tgt.src + test.src-tgt.tgt
        with open(os.path.join(save_dir, f"{src}{tgt}", f"test.{src}-{tgt}.{src}"), 'w') as src_f, open(os.path.join(save_dir, f"{src}{tgt}", f"test.{src}-{tgt}.{tgt}"), 'w') as tgt_f:
            for i in range(len(src_devtest)):
                src_f.write(src_devtest[i] + '\n')
                tgt_f.write(tgt_devtest[i] + '\n')
                

        with open(os.path.join(save_dir, f"{src}{tgt}", f"test.{src}-{tgt}.json"), 'w') as f:
            translation_dicts = []
            for i in range(len(src_devtest)):
                translation_dicts.append({"translation": {
                    src: src_devtest[i],
                    tgt: src_devtest[i]
                }})

            json.dump(translation_dicts, f)
        

        

    # # Read and concatenate the Flores dataset files
    # df = pd.DataFrame()
    # for sd in SUBDIRS:
    #     ndf = pd.DataFrame()
    #     for l in LANG_IDS:
    #         filename = os.path.join(FLORES_DIR, sd, f'{sd}.{l}')
    #         with open(filename) as f:
    #             ndf[l] = f.readlines()
    #     df = pd.concat([df, ndf])

    # # Generate the required files for each language pair
    # for idx1, idx2 in permutations(range(len(OPUS_LANGUAGE_IDS)), 2):
    #     combo_dir = os.path.join(save_dir, f"{OPUS_LANGUAGE_IDS[idx1]}{OPUS_LANGUAGE_IDS[idx2]}")
    #     os.makedirs(combo_dir, exist_ok=True)

    #     with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.{OPUS_LANGUAGE_IDS[idx1]}"), 'w') as f:
    #         f.writelines(df[LANG_IDS[idx1]])

    #     with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.{OPUS_LANGUAGE_IDS[idx2]}"), 'w') as f:
    #         f.writelines(df[LANG_IDS[idx2]])

    #     with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.json"), 'w') as f:
    #         json.dump([
    #             {
    #                 "translation": {
    #                     OPUS_LANGUAGE_IDS[idx1]: row[LANG_IDS[idx1]],
    #                     OPUS_LANGUAGE_IDS[idx2]: row[LANG_IDS[idx2]]
    #                 }
    #             }
    #             for _, row in df.iterrows()
    #         ], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data from Flores dataset.")
    parser.add_argument('save_dir', type=str, help="Directory to save the generated files")
    args = parser.parse_args()

    main(args.save_dir)