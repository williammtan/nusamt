
from collections import defaultdict
from datasets import load_dataset
import argparse
import pandas as pd
import numpy as np
import subprocess
import tempfile
import logging
import shutil
import gzip
import json
import os


# CONSTANTS
LANGUAGE_NAMES = ['Balinese', 'Acehnese', 'Banjar', 'Buginese', 'Minangkabau', 'Sundanese', 'Javanese', 'Indonesian', 'English']
OPUS_LANGUAGE_IDS = ['ban', 'ace', 'bjn', 'bug', 'min', 'su', 'jv', 'id', 'en']
NLLB_LANGUAGE_IDS = ['ban_Latn', 'ace_Latn', 'bjn_Latn', 'bug_Latn', 'min_Latn', 'sun_Latn', 'jav_Latn', 'ind_Latn', 'eng_Latn']

# ENVIRONMENT SETTINGS
CACHE_DIR = os.environ.get("CACHE_DIR") or ".cache"
PREPROCESS_CACHE_DIR = os.path.join(CACHE_DIR, "preprocess")

logging.basicConfig(level=logging.INFO)

def make_dir_if_not_exists(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def replace_multiple(string, dic):
    for k,v in dic.items():
        string = string.replace(k, v)
    return string

def sort_pair(pair):
    return tuple(sorted(pair))

def get_lang_directions(languages, base_languages, order_matters=True, skip_base_pairs=False):
    """Helper function to generate the language directions."""

    directions = []
    
    for base in base_languages:
        if skip_base_pairs:
            other_languages = [lang for lang in languages if lang not in base_languages]
        else:
            other_languages = [lang for lang in languages if lang != base]

        for lang in other_languages:

            if order_matters:
                directions.append((base, lang))
                directions.append((lang, base))
            else:
                pair = tuple(sorted([base, lang]))
                if pair not in directions:
                    directions.append(pair)
    
    return directions

def temp_extract_blob_from_git(url) -> tempfile.TemporaryDirectory:
    tmpdir = tempfile.TemporaryDirectory()
    subprocess.run(["git", "clone", url, tmpdir.name, "--quiet"])
    logging.debug(f"Cloned from {url}")
    return tmpdir

def download_nusax(overwrite=True):
    NUSAX_COLUMNS = ["", "id", "ace", "bjn", "en", "", "", "su", "ban", "bug", "jv", "min", ""]

    nusax_dir = os.path.join(PREPROCESS_CACHE_DIR, "nusax")
    if os.path.isdir(nusax_dir):
        if overwrite:
            shutil.rmtree(nusax_dir)
        else:
            logging.info("Skipping downloading NusaX, cache dir found.")
            return
    os.makedirs(nusax_dir, exist_ok=True)

    tmpdir = temp_extract_blob_from_git("https://github.com/IndoNLP/nusax")
    datadir = os.path.join(tmpdir.name, "datasets/mt/")
    df = pd.concat([
        pd.read_csv(os.path.join(datadir, "train.csv")),
        pd.read_csv(os.path.join(datadir, "valid.csv")),
        pd.read_csv(os.path.join(datadir, "test.csv"))
    ])
    logging.info(f"Writing {len(df)} sentences for {len(df.columns)} langauges from NusaX.")

    for lang in OPUS_LANGUAGE_IDS:
        if lang in NUSAX_COLUMNS:
            texts = df[df.columns[NUSAX_COLUMNS.index(lang)]]
        else:
            texts = []
        with gzip.open(os.path.join(nusax_dir, f"{lang}.gz"), "wt") as out:
            logging.debug(f"Writing {len(texts)} sentences for {lang}")
            out.write('\n'.join(texts))
    
    tmpdir.cleanup()
    return nusax_dir

def download_seed(overwrite=True):
    NLLB_SEED_LANGS = ["ban", "bjn", "ace", "bug", "en"]

    seed_dir = os.path.join(PREPROCESS_CACHE_DIR, "seed")
    if os.path.isdir(seed_dir):
        if overwrite:
            shutil.rmtree(seed_dir)
        else:
            logging.info("Skipping downloading NLLB Seed, cache dir found.")
            return
    os.makedirs(seed_dir, exist_ok=True)

    tmpdir = temp_extract_blob_from_git("https://github.com/openlanguagedata/seed")
    datadir = os.path.join(tmpdir.name, "seed")

    for lang in OPUS_LANGUAGE_IDS:
        nllb_id = NLLB_LANGUAGE_IDS[OPUS_LANGUAGE_IDS.index(lang)]
        if lang in NLLB_SEED_LANGS:
            with open(os.path.join(datadir, nllb_id), 'r') as f:
                data = f.read()
        else:
            data = ""
        
        logging.debug("Writing {} sentences for {}".format(len(data.split('\n'))-1, lang))
        
        with gzip.open(os.path.join(seed_dir, f"{lang}.gz"), "wt") as out:
            out.write(data)

    tmpdir.cleanup()
    return seed_dir

def download_nusa_writes(overwrite=True):
    NUSA_WRITES_LANGS = ["", "", "", "", "min", "", "jav", "", ""]

    nusa_dir = os.path.join(PREPROCESS_CACHE_DIR, "nusa_writes")
    if os.path.isdir(nusa_dir):
        if overwrite:
            shutil.rmtree(nusa_dir)
        else:
            logging.info("Skipping downloading nusa-writes, cache dir found.")
            return
    os.makedirs(nusa_dir, exist_ok=True)
    tmpdir = temp_extract_blob_from_git("https://github.com/IndoNLP/nusa-writes")
    datadir = os.path.join(tmpdir.name, "data")

    data_dict = defaultdict(list)

    for lang1, lang2 in get_lang_directions(OPUS_LANGUAGE_IDS, OPUS_LANGUAGE_IDS, order_matters=False):
        data_dict[lang1 + "-" + lang2] = []
    

    for lang_nw in NUSA_WRITES_LANGS:
        if lang_nw == "":
            continue
        lang_idx = NUSA_WRITES_LANGS.index(lang_nw)
        lang = OPUS_LANGUAGE_IDS[lang_idx]

        lang1, lang2 = sort_pair(["id", lang])

        for partition in ["train", "valid", "test"]:
            filename = os.path.join(datadir, f"nusa_kalimat-mt-{lang_nw}-{partition}.csv")
            df = pd.read_csv(filename)
            # cleanse new lines
            df[df.columns[-1]] = df[df.columns[-1]].str.replace('\n', '')
            df[df.columns[-2]] = df[df.columns[-1]].str.replace('\n', '')

            values = df[df.columns[-2:]].values.tolist()

            if (lang1, lang2) != ("id", "lang"):
                # swap
                values = [[sent2, sent1] for sent1, sent2 in values]

            data_dict[lang1+'-'+lang2].extend(df[df.columns[-2:]].values.tolist())

    for lang_pair, examples in data_dict.items():
        lang1, lang2 = lang_pair.split('-')
        base_filepath = os.path.join(nusa_dir, lang_pair) # eg. nusa_writes/ban-en.en
        with gzip.open(base_filepath + "." + lang1 + ".gz", 'wt') as f1, gzip.open(base_filepath + "." + lang2 + ".gz", 'wt') as f2:
            for sent1, sent2 in examples:
                f1.write(sent1 + '\n')
                f2.write(sent2 + '\n')
        
        if len(examples) > 0:
            logging.debug(f"Wrote {len(examples)} sentence pairs in the {lang1}-{lang2} direction.")
    
    tmpdir.cleanup()
    return nusa_dir

def write_template(idx1, idx2, template_yaml_path, output_dir, configs_dir): # alr /opus and dir created
    make_dir_if_not_exists(configs_dir)

    with open(template_yaml_path) as f:
        template = f.read()
    
    template = replace_multiple(template, {
        "{{source}}": OPUS_LANGUAGE_IDS[idx1],
        "{{target}}": OPUS_LANGUAGE_IDS[idx2],
        "{{nllb_source}}": NLLB_LANGUAGE_IDS[idx1],
        "{{nllb_target}}": NLLB_LANGUAGE_IDS[idx2],
        "{{output_directory}}": output_dir,
        "{{cache_dir}}": os.path.abspath(PREPROCESS_CACHE_DIR)
    })

    config_filepath = os.path.join(configs_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}_pipeline.yaml")
    with open(config_filepath, 'w') as f:
        f.write(template)
    
    return config_filepath

def create_alignment_files(idx1, idx2, output_dir):
    """Create empty alignment files for a given language pair"""
    DEFAULT_ALIGNMENT_FILES = [
        "NLLB_latest_xml_{target}-{source}.xml.gz",
        "WikiMatrix_v1_xml_{target}-{source}.xml.gz",
        "wikimedia_latest_xml_{target}-{source}.xml.gz",
        "NLLB_latest_xml_{source}-{target}.xml.gz",
        "WikiMatrix_v1_xml_{source}-{target}.xml.gz",
        "wikimedia_latest_xml_{source}-{target}.xml.gz"
    ]
    make_dir_if_not_exists(output_dir)

    for filename in DEFAULT_ALIGNMENT_FILES:
        alignment_filepath = os.path.join(output_dir, filename.format(source=OPUS_LANGUAGE_IDS[idx1], target=OPUS_LANGUAGE_IDS[idx2]))
        if not os.path.isfile(alignment_filepath):
            # Fill with empty
            f = open(alignment_filepath, 'x')
            f.close()

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

def download_nllb(idx1: int, idx2: int, laser_score_threshold: float, lid_score_threshold: float, max_size: int, output_dir: str):
    # Download from allenai/nllb huggingface dataset
    dataset = load_nllb_dataset_allenai(NLLB_LANGUAGE_IDS[idx1], NLLB_LANGUAGE_IDS[idx2])

    # Filter by laser score and lid scores
    translations = []
    laser_scores = []
    for example in dataset:
        if example["laser_score"] >= laser_score_threshold and example["source_sentence_lid"] > lid_score_threshold and example["target_sentence_lid"] > lid_score_threshold:
            translations.append(example["translation"])
            laser_scores.append(example["laser_score"])

    sorted_indices = np.argsort(laser_scores)
    translations = np.array(translations)[sorted_indices]
    translations = translations[:max_size]

    # Save to output_dir
    with gzip.open(os.path.join(output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx1]}.gz"), 'wt') as src_out:
        src_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx1]] for t in translations]))
    with gzip.open(os.path.join(output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx2]}.gz"), 'wt') as tgt_out:
        tgt_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx2]] for t in translations]))

def process_filtered_files(idx1, idx2, output_dir, clean_output_dir):
    for (source, target) in [(OPUS_LANGUAGE_IDS[idx1], OPUS_LANGUAGE_IDS[idx2]), (OPUS_LANGUAGE_IDS[idx2], OPUS_LANGUAGE_IDS[idx1])]:
        direction_output_dir = os.path.join(clean_output_dir, f"{source}{target}")
        os.makedirs(direction_output_dir, exist_ok=True)

        for subset in ["train", "test", "valid"]:
            bitext_list = []

            with gzip.open(os.path.join(output_dir, f"{source}-{target}.{source}.{subset}.gz"), 'rt') as src_file, gzip.open(os.path.join(output_dir, f"{source}-{target}.{target}.{subset}.gz"), 'rt') as tgt_file:
                with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{source}"), 'w') as src_outfile, open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{target}"), 'w') as tgt_outfile:
                    for src_text, tgt_text in zip(src_file, tgt_file):
                        bitext_list.append({
                            "translation": {
                                source: src_text.strip("\n"),
                                target: tgt_text.strip("\n")
                            }
                        })
                        src_outfile.write(src_text)
                        tgt_outfile.write(tgt_text)

            with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.json"), 'w') as j:
                json.dump(bitext_list, j)
    
    return direction_output_dir

def main(args):
    download_nusax(overwrite=args.overwrite_cache)
    download_seed(overwrite=args.overwrite_cache)
    download_nusa_writes(overwrite=args.overwrite_cache)

    opus_dir = os.path.join(args.output_dir, "opus")
    clean_output_dir = os.path.join(args.output_dir, "clean")
    os.makedirs(opus_dir, exist_ok=True)

    for lang1, lang2 in args.language_pairs:
        idx1 = OPUS_LANGUAGE_IDS.index(lang1)
        idx2 = OPUS_LANGUAGE_IDS.index(lang2)

        run_output_dir = os.path.join(opus_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}")
        make_dir_if_not_exists(run_output_dir)

        config_filepath = write_template(idx1, idx2, args.template_yaml, run_output_dir, args.configs_dir)
        create_alignment_files(idx1, idx2, run_output_dir)

        download_nllb(idx1, idx2, args.laser_threshold, args.lid_threshold, args.max_nllb_size, run_output_dir)

        # Execute and wait pipeline execution
        opus_command = ["opusfilter", config_filepath]
        if args.overwrite_opus:
            opus_command.append("--overwrite")
        subprocess.run(opus_command, env=os.environ.update({"PYTHONPATH": os.getcwd()}), check=True)
        
        process_filtered_files(idx1, idx2, run_output_dir, clean_output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline for specified language pairs.")

    # Required args
    parser.add_argument(
        "--language_pairs",
        type=str,
        required=True,
        help="Comma-separated list of language pairs in OPUS format (e.g., en-ban,ban-en,id-ban,ban-id)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    

    # Not required args
    parser.add_argument(
        "--laser_threshold",
        type=float,
        default=1.07,
        help="Laser score threshold"
    )
    parser.add_argument(
        "--lid_threshold",
        type=float,
        default=0.95,
        help="LID score threshold"
    )
    parser.add_argument(
        "--template_yaml",
        type=str,
        default="configs/pipeline_template.yaml",
        help="Path to the pipeline template YAML file"
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs/auto_pipeline/",
        help="Directory to store generated config files"
    )
    parser.add_argument(
        "--max_nllb_size",
        type=int,
        default=1000000,
        help="Maximum size for NLLB dataset"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite existing cache files"
    )
    parser.add_argument(
        "--overwrite_opus",
        action="store_true",
        help="Overwrite opus files"
    )

    args = parser.parse_args()

    args.language_pairs = [
        sort_pair(pair.split('-'))
        for pair in args.language_pairs.split(',')
    ]

    main(args)