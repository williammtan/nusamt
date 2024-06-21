# pipeline.py
# A pipeline that will automatically run the pipeline.yaml script for all combinations of language pairs

from itertools import combinations
from datasets import load_dataset
import pandas as pd
import numpy as np
import subprocess
import shutil
import gzip
import glob
import json
import os

TEST = True


LANGUAGE_NAMES = ['Balinese', 'Acehnese', 'Banjar', 'Buginese', 'Minangkabau', 'Sundanese', 'Javanese', 'Indonesian', 'English']
OPUS_LANGUAGE_IDS = ['ban', 'ace', 'bjn', 'bug', 'min', 'su', 'jv', 'id', 'en']
NLLB_LANGUAGE_IDS = ['ban_Latn', 'ace_Latn', 'bjn_Latn', 'bug_Latn', 'min_Latn', 'sun_Latn', 'jav_Latn', 'ind_Latn', 'eng_Latn']

# if TEST:
#     LANGUAGE_NAMES = ["Sundanese", "Minangkabau", "Buginese"]
#     OPUS_LANGUAGE_IDS = ["su", "min", "bug"]
#     NLLB_LANGUAGE_IDS = [
#         "sun_Latn",
#         "min_Latn",
#         "bug_Latn"
#     ]



SKIP_DIRECTIONS = [("id", "bug"), ("bug", "id")]

TEMPLATE_YAML = "configs/pipeline_template.yaml"
CONFIGS_DIR = "configs/auto_pipeline/"
DEFAULT_ALIGNMENT_FILES = [
    "NLLB_latest_xml_{target}-{source}.xml.gz",
    "WikiMatrix_v1_xml_{target}-{source}.xml.gz",
    "wikimedia_latest_xml_{target}-{source}.xml.gz",
    "NLLB_latest_xml_{source}-{target}.xml.gz",
    "WikiMatrix_v1_xml_{source}-{target}.xml.gz",
    "wikimedia_latest_xml_{source}-{target}.xml.gz"
]
OUTPUT_DIR = "out-pipeline/"
NUSAX_DIR = "/workspace/nusax/datasets/mt/"
NUSAX_COLUMNS = ["", "id", "ace", "ban", "en", "", "", "su", "ban", "bug", "jv", "min"]
NLLB_SEED_DIR = "/workspace/seed/seed/"
NLLB_SEED_LANGS = ["ban", "bjn", "ace", "bug", "en"]
MAX_NLLB_SIZE = 1000000
HUGGINGFACE_CACHE_DIR = "/root/.cache/huggingface/datasets/"

with open(TEMPLATE_YAML) as f:
    default_template_str = f.read()

os.makedirs(CONFIGS_DIR, exist_ok=True)

# convert seed and nusax to gzip
nusax_df = pd.concat([
    pd.read_csv(os.path.join(NUSAX_DIR, "train.csv")),
    pd.read_csv(os.path.join(NUSAX_DIR, "valid.csv")),
    pd.read_csv(os.path.join(NUSAX_DIR, "test.csv"))
]) # columns: indonesian,acehnese,banjarese,english,madurese,ngaju,sundanese,balinese,buginese,javanese,minangkabau,toba_batak

os.makedirs(os.path.join(OUTPUT_DIR, "nusax"), exist_ok=True)
for lang in OPUS_LANGUAGE_IDS:
    if lang in NUSAX_COLUMNS:
        texts = nusax_df[nusax_df.columns[NUSAX_COLUMNS.index(lang)]]
    else:
        texts = []
    with gzip.open(os.path.join(OUTPUT_DIR, "nusax", f"{lang}.gz"), "wt") as out:
            out.write('\n'.join(texts))

os.makedirs(os.path.join(OUTPUT_DIR, "seed"), exist_ok=True)
for lang in OPUS_LANGUAGE_IDS:
    nllb_id = NLLB_LANGUAGE_IDS[OPUS_LANGUAGE_IDS.index(lang)]
    if lang in NLLB_SEED_LANGS:
        with open(os.path.join(NLLB_SEED_DIR, nllb_id)) as seed:
            data = seed.read()
    else:
        data = ""
    
    with gzip.open(os.path.join(OUTPUT_DIR, "seed", f"{lang}.gz"), "wt") as out:
        out.write(data)

for idx1, idx2 in combinations(range(len(LANGUAGE_NAMES)), 2):
# for idx1, idx2 in [(3,4), (3,5)]:
    clean_output_dir = os.path.join(OUTPUT_DIR, 'clean')

    if (idx1,idx2) in SKIP_DIRECTIONS or os.path.exists(os.path.join(clean_output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}")):
        print("skipped", f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}")
        continue
    # 1. Overwrite default template 

    template = default_template_str
    template = template.replace("{{source}}", OPUS_LANGUAGE_IDS[idx1])
    template = template.replace("{{target}}", OPUS_LANGUAGE_IDS[idx2])
    template = template.replace("{{nllb_source}}", NLLB_LANGUAGE_IDS[idx1])
    template = template.replace("{{nllb_target}}", NLLB_LANGUAGE_IDS[idx2])

    run_output_dir = os.path.join(OUTPUT_DIR, 'opus', f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}")
    template = template.replace("{{output_directory}}", run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)
    
    config_filepath = os.path.join(CONFIGS_DIR, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}_pipeline.yaml")

    with open(config_filepath, 'w') as f:
        f.write(template)
    
    # 2. Prepare empty output files for the downloads & load NLLB data from AllenAI
    for filename in DEFAULT_ALIGNMENT_FILES:
        alignment_filepath = os.path.join(run_output_dir, filename.format(source=OPUS_LANGUAGE_IDS[idx1], target=OPUS_LANGUAGE_IDS[idx2]))
        if not os.path.isfile(alignment_filepath):
            f = open(alignment_filepath, 'x')
            f.close()

    try:
        nllb_dataset = load_dataset("allenai/nllb", f"{NLLB_LANGUAGE_IDS[idx1]}-{NLLB_LANGUAGE_IDS[idx2]}", ignore_verifications=True)["train"]
        nllb_dataset_source = NLLB_LANGUAGE_IDS[idx1]
        nllb_dataset_target = NLLB_LANGUAGE_IDS[idx2]
    except ValueError:
        try:
            nllb_dataset = load_dataset("allenai/nllb", f"{NLLB_LANGUAGE_IDS[idx2]}-{NLLB_LANGUAGE_IDS[idx1]}", ignore_verifications=True)["train"]
            nllb_dataset_source = NLLB_LANGUAGE_IDS[idx2]
            nllb_dataset_target = NLLB_LANGUAGE_IDS[idx1]
        except ValueError:
            nllb_dataset_source = ""
            nllb_dataset_target = ""
            nllb_dataset = []
    
    scores = np.array([example["laser_score"] for example in nllb_dataset])
    sorted_indices = np.argsort(scores)
    translations = np.array([example["translation"] for example in nllb_dataset])[sorted_indices]
    translations = translations[:MAX_NLLB_SIZE]

    # dump to nllb gz
    with gzip.open(os.path.join(run_output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx1]}.gz"), 'wt') as src_out, gzip.open(os.path.join(run_output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx2]}.gz"), 'wt') as tgt_out:
        src_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx1]] for t in translations]))
        tgt_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx2]] for t in translations]))
    
    # 3. Execute pipeline
    subprocess.run(["opusfilter", config_filepath], env=os.environ.update({"PYTHONPATH": os.getcwd()}))

    # 4. Process filtered files
    
    for (source, target) in [(OPUS_LANGUAGE_IDS[idx1], OPUS_LANGUAGE_IDS[idx2]), (OPUS_LANGUAGE_IDS[idx2], OPUS_LANGUAGE_IDS[idx1])]:
        direction_output_dir = os.path.join(clean_output_dir, f"{source}-{target}")
        os.makedirs(direction_output_dir, exist_ok=True)

        for subset in ["train", "test", "valid"]:
            bitext_list = []

            with gzip.open(os.path.join(run_output_dir, f"{source}-{target}.{source}.{subset}.gz"), 'rt') as src_file, gzip.open(os.path.join(run_output_dir, f"{source}-{target}.{target}.{subset}.gz"), 'rt') as tgt_file:
                with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{source}"), 'w') as src_outfile, open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{target}"), 'w') as tgt_outfile:
                    for src_text, tgt_text in zip(src_file, tgt_file):
                        bitext_list.append({
                            "translation": {
                                source: src_text,
                                target: tgt_text
                            }
                        })
                        src_outfile.write(src_text)
                        tgt_outfile.write(tgt_text)

            with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.json"), 'w') as j:
                json.dump(bitext_list, j)
    
    # Clean-up
    # os.remove(os.path.join(run_output_dir, f"NLLB_latest_raw_{OPUS_LANGUAGE_IDS[idx1]}.zip"))
    # os.remove(os.path.join(run_output_dir, f"NLLB_latest_raw_{OPUS_LANGUAGE_IDS[idx2]}.zip"))
    shutil.rmtree(run_output_dir)

    # if os.path.exists(HUGGINGFACE_CACHE_DIR):
    shutil.rmtree(HUGGINGFACE_CACHE_DIR)


    # convert {target}-{source}.{target/source}.{train/test/valid}.gz to {train/test/valid}.{target}-{source}.{target/source} & {train/test/valid}.{target}-{source}.json

