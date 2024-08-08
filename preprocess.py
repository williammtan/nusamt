from collections import defaultdict
from itertools import combinations
import subprocess
import argparse
import tempfile
import logging
import shutil
import gzip
import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import fasttext
import sqlite3
from hashlib import sha256

# CONSTANTS
LANGUAGE_NAMES = ['Balinese', 'Acehnese', 'Banjar', 'Buginese', 'Minangkabau', 'Sundanese', 'Javanese', 'Indonesian', 'English']
OPUS_LANGUAGE_IDS = ['ban', 'ace', 'bjn', 'bug', 'min', 'su', 'jv', 'id', 'en']
NLLB_LANGUAGE_IDS = ['ban_Latn', 'ace_Latn', 'bjn_Latn', 'bug_Latn', 'min_Latn', 'sun_Latn', 'jav_Latn', 'ind_Latn', 'eng_Latn']

LLAMA3_SYSTEM_PROMPT="Clean the data by identifying and fixing problems in parallel sentences. The problems include misalignment, repetition, incomplete translations, and inconsistent formatting. Provide the cleaned output without any repetition."

# ENVIRONMENT SETTINGS
CACHE_DIR = os.environ.get("CACHE_DIR") or ".cache"
PREPROCESS_CACHE_DIR = os.path.join(CACHE_DIR, "preprocess")

# GLOBAL OBJECTS
CACHE_MANAGER = None

logging.basicConfig(level=logging.DEBUG)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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

def download_minangnlp(overwrite=True):
    partitions = ["train", "test", "validation"]
    minangnlp_dir = os.path.join(PREPROCESS_CACHE_DIR, "minangnlp")

    if os.path.isdir(minangnlp_dir):
        if overwrite:
            shutil.rmtree(minangnlp_dir)
        else:
            logging.info("Skipping downloading NusaX, cache dir found.")
            return
    os.makedirs(minangnlp_dir, exist_ok=True)

    ds = load_dataset("SEACrowd/minangnlp_mt")
    src_sentences = []
    tgt_sentences = []
    for p in partitions:
        for example in ds[p]:
            src_sentences.append(example['src'].replace('\n', ''))
            tgt_sentences.append(example['tgt'].replace('\n', ''))
    src = "id"
    tgt = "min"

    for lang1, lang2 in get_lang_directions(OPUS_LANGUAGE_IDS, OPUS_LANGUAGE_IDS, order_matters=False):
        with  gzip.open(os.path.join(minangnlp_dir, f"{lang1}-{lang2}.{lang1}.gz"), "wt", encoding="utf-8") as out1, gzip.open(os.path.join(minangnlp_dir, f"{lang1}-{lang2}.{lang2}.gz"), "wt", encoding="utf-8") as out2:
            if lang1 == src and lang2 == tgt:
                out1.write('\n'.join(src_sentences))
                out2.write('\n'.join(tgt_sentences))

def download_indonesianmnt(overwrite=True):
    indonesianmnt_dir = os.path.join(PREPROCESS_CACHE_DIR, "indonesianmnt")

    if os.path.isdir(indonesianmnt_dir):
        if overwrite:
            shutil.rmtree(indonesianmnt_dir)
        else:
            logging.info("Skipping downloading NusaX, cache dir found.")
            return
    os.makedirs(indonesianmnt_dir, exist_ok=True)

    df = pd.read_csv('https://huggingface.co/datasets/Exqrch/IndonesianNMT/raw/main/id-min.tsv', sep='\t', header=0)
    df = df.dropna()
    src = "id"
    src_column = "Indonesian"
    tgt = "min"
    tgt_column = "Minangkabau"

    for lang1, lang2 in get_lang_directions(OPUS_LANGUAGE_IDS, OPUS_LANGUAGE_IDS, order_matters=False):
        with gzip.open(os.path.join(indonesianmnt_dir, f"{lang1}-{lang2}.{lang1}.gz"), "wt", encoding="utf-8") as out1, gzip.open(os.path.join(indonesianmnt_dir, f"{lang1}-{lang2}.{lang2}.gz"), "wt", encoding="utf-8") as out2:
            if lang1 == src and lang2 == tgt:
                out1.write('\n'.join(df[src_column]))
                out2.write('\n'.join(tgt_column))
            


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
    ]).dropna()
    logging.info(f"Writing {len(df)} sentences for {len(df.columns)} langauges from NusaX.")

    for lang in OPUS_LANGUAGE_IDS:
        if lang in NUSAX_COLUMNS:
            texts = df[df.columns[NUSAX_COLUMNS.index(lang)]].apply(lambda x: x.replace('\n', ''))
        else:
            texts = []
        with gzip.open(os.path.join(nusax_dir, f"{lang}.gz"), "wt", encoding="utf-8") as out:
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

    data_dict = defaultdict(list)

    for lang1, lang2 in get_lang_directions(OPUS_LANGUAGE_IDS, OPUS_LANGUAGE_IDS, order_matters=False):
        if lang1 in NLLB_SEED_LANGS and lang2 in NLLB_SEED_LANGS:
            nllb_id1 = NLLB_LANGUAGE_IDS[OPUS_LANGUAGE_IDS.index(lang1)]
            nllb_id2 = NLLB_LANGUAGE_IDS[OPUS_LANGUAGE_IDS.index(lang2)]
            with open(os.path.join(datadir, nllb_id1), 'r', encoding="utf-8") as f1, open(os.path.join(datadir, nllb_id2), 'r', encoding="utf-8") as f2:
                sents1 = f1.read().split('\n')
                sents2 = f2.read().split('\n')
                data_dict[lang1 + "-" + lang2] = list(zip(sents1, sents2))
        else:
            data_dict[lang1 + "-" + lang2] = []
    
    for direction, sentences in data_dict.items():
        lang1, lang2 = direction.split('-')
        with gzip.open(os.path.join(seed_dir, f"{lang1}-{lang2}.{lang1}.gz"), "wt", encoding="utf-8") as out1, gzip.open(os.path.join(seed_dir, f"{lang1}-{lang2}.{lang2}.gz"), "wt", encoding="utf-8") as out2:
            for sent1, sent2 in sentences:
                out1.write(sent1+"\n")
                out2.write(sent2+"\n")

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
        with gzip.open(base_filepath + "." + lang1 + ".gz", 'wt', encoding="utf-8") as f1, gzip.open(base_filepath + "." + lang2 + ".gz", 'wt', encoding="utf-8") as f2:
            for sent1, sent2 in examples:
                f1.write(sent1 + '\n')
                f2.write(sent2 + '\n')
        
        if len(examples) > 0:
            logging.debug(f"Wrote {len(examples)} sentence pairs in the {lang1}-{lang2} direction.")
    
    tmpdir.cleanup()
    return nusa_dir

def write_template(idx1, idx2, template_yaml_path, output_dir, configs_dir): # alr /opus and dir created
    make_dir_if_not_exists(configs_dir)

    with open(template_yaml_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    template = replace_multiple(template, {
        "{{source}}": OPUS_LANGUAGE_IDS[idx1],
        "{{target}}": OPUS_LANGUAGE_IDS[idx2],
        "{{source_fullname}}": LANGUAGE_NAMES[idx1],
        "{{target_fullname}}": LANGUAGE_NAMES[idx2],
        "{{nllb_source}}": NLLB_LANGUAGE_IDS[idx1],
        "{{nllb_target}}": NLLB_LANGUAGE_IDS[idx2],
        "{{output_directory}}": output_dir,
        "{{cache_dir}}": os.path.abspath(PREPROCESS_CACHE_DIR)
    })

    config_filepath = os.path.join(configs_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}_pipeline.yaml")
    with open(config_filepath, 'w', encoding='utf-8') as f:
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
            f = open(alignment_filepath, 'x', encoding='utf-8')
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


class SqliteCacheManager:
    def __init__(self, database_path=os.path.join(PREPROCESS_CACHE_DIR, "cache.db")):
    
        # initialize connection to db
        self.con = sqlite3.connect(database_path)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()


class CacheTable:
    def __init__(self, name: str, columns, hash_columns):
        """
        Required functionality:
            - Create table if not exists (DONE)
            - Get cache if exists
            - Batch write new cache
        """
        self.name = name
        self.columns = {"hash": "TEXT PRIMARY KEY"}
        self.columns.update(columns)
        self.hash_columns = hash_columns # list of columns to use as hash for


        # initialize table
        self.create_table()

    def create_table(self):
        """Initializes the cache table if it doesn't exist"""

        column_string = ",\n".join([
                col_name + " " + descriptor
            for col_name, descriptor in self.columns.items()
        ])
        sql_create_table = f"""CREATE TABLE IF NOT EXISTS {self.name} ({column_string});"""
        CACHE_MANAGER.cur.execute(sql_create_table)
        CACHE_MANAGER.con.commit()
    
    def hash_data(self, data):
        hashed_data_str = json.dumps({
            col: data[col]
            for col in self.hash_columns
        }, sort_keys=True)
        return sha256(hashed_data_str.encode('utf-8')).hexdigest()
    
    def get(self, data):
        """Find and returns the given row by hash if it exists"""
        data_hash = self.hash_data(data)
        res = CACHE_MANAGER.cur.execute(f"SELECT * FROM {self.name} WHERE hash = \"{data_hash}\"")
        return res.fetchone()

    def batch_update(self, datas):
        """Adds a multiple new rows"""
        data_lists = []
        for data in datas:
            d_list = [self.hash_data(data)]
            for col in list(self.columns.keys())[1:]:
                d_list.append(data[col])
            data_lists.append(d_list)
        
        sql_insert = f"INSERT OR IGNORE INTO {self.name}({', '.join(self.columns)}) VALUES({', '.join(['?' for _ in self.columns.keys()])})"
        CACHE_MANAGER.cur.executemany(sql_insert, data_lists)
        CACHE_MANAGER.con.commit()
    

class Cleaner:
    def __init__(self, model_path, system_prompt=LLAMA3_SYSTEM_PROMPT, temperature=0, n_gpus=4):
        self.model = LLM(model=model_path, tensor_parallel_size=n_gpus)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.system_prompt = system_prompt
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)

        self.cache_table = CacheTable("cleaner", {
            "source_lang": "TEXT CHECK(length(source_lang) <= 8)",
            "target_lang": "TEXT CHECK(length(target_lang) <= 8)",
            "source_sentence": "TEXT",
            "target_sentence" : "TEXT",
            "cleaned_source_sentence": "TEXT",
            "cleaned_target_sentence": "TEXT"
        }, ["source_lang", "target_lang", "source_sentence", "target_sentence"])
    
    def _format_prompt(self, translation, lang1, lang2):
        lang1_name = LANGUAGE_NAMES[NLLB_LANGUAGE_IDS.index(lang1)]
        lang2_name = LANGUAGE_NAMES[NLLB_LANGUAGE_IDS.index(lang2)]
        user_prompt = f"{lang1_name}: {translation[lang1]}\n{lang2_name}: {translation[lang2]}"
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[:-1]

    def _parse_response(self, response, lang1, lang2):
        """Parse a response given the source and target languages, returning a translation dict."""

        lang1_name = LANGUAGE_NAMES[NLLB_LANGUAGE_IDS.index(lang1)] # TODO: evaluate if necessary to retry requests if responses cannot be parsed
        lang2_name = LANGUAGE_NAMES[NLLB_LANGUAGE_IDS.index(lang2)]

        response = response.replace('<|eot_id|>', '') # Remove end token for llama3

        try:
            source_sentence, target_sentence = response.split(f"\n{lang2_name}:")
            source_sentence = source_sentence.replace(f"{lang1_name}: ", "").strip()
            target_sentence = target_sentence.strip()
        except ValueError:
            print(f"Unparsable response: \"{repr(response)}\", response will be saved as a empty string")
            source_sentence = ""
            target_sentence = ""

        return {
            lang1: source_sentence,
            lang2: target_sentence
        }
    
    def predict_batch(self, translations):
        """Cleans a list of translation dicts and returns a list of cleaned translation dicts."""

        formated_prompts = []
        language_pairs = []
        cleaned_translations = [None] * len(translations)
        uncached_indices = []

        for i, translation in enumerate(translations):
            lang1, lang2 = translation.keys()
            row = self.cache_table.get({
                "source_lang": lang1,
                "target_lang": lang2,
                "source_sentence": translation[lang1],
                "target_sentence": translation[lang2]
            })
            if row:
                # cached entry exists
                cleaned_translations[i] = {
                    lang1: row["cleaned_source_sentence"],
                    lang2: row["cleaned_target_sentence"]
                }
            else:
                uncached_indices.append(i)
                formated_prompts.append(self._format_prompt(translation, lang1, lang2))
                language_pairs.append((lang1, lang2))
        
        results = self.model.generate(formated_prompts, self.sampling_params)
        cleaned_responses = [
            output.outputs[0].text
            for output in results
        ]
        
        # Parse the results and convert back to translations
        new_cleaned_translations = []
        for i,idx in enumerate(uncached_indices):
            t = self._parse_response(
                    response=cleaned_responses[i], 
                    lang1=language_pairs[i][0],
                    lang2=language_pairs[i][1]
                    )
            if t[lang1] == "" and t[lang2] == "":
                t = translations[i]
            # t[lang1+"_ori"] = translations[i][lang1]
            # t[lang2+"_ori"] = translations[i][lang2]
            new_cleaned_translations.append({
                "source_lang": language_pairs[i][0],
                "target_lang": language_pairs[i][1],
                "source_sentence": translations[i][lang1],
                "target_sentence": translations[i][lang2],
                "cleaned_source_sentence": t[lang1],
                "cleaned_target_sentence": t[lang2]
            })
            cleaned_translations[idx] = t
        
        self.cache_table.batch_update(new_cleaned_translations)
        
        return cleaned_translations

def download_nllb(idx1: int, idx2: int, cleaner: Cleaner, lid_model: str, laser_score_threshold: float, lid_score_threshold: float, max_size: int, output_dir: str):
    # Download from allenai/nllb huggingface dataset
    dataset = load_nllb_dataset_allenai(NLLB_LANGUAGE_IDS[idx1], NLLB_LANGUAGE_IDS[idx2])

    # Initial filter by LASER score
    cleaned_dataset = []
    for example in dataset:
        if example["laser_score"] > laser_score_threshold:
            if "alkitab.mobi" not in example["source_sentence_url"] and "alkitab.mobi" not in example["target_sentence_url"]:
                cleaned_dataset.append(example)

    # Load LID model and run predictions
    model_path = hf_hub_download(repo_id=lid_model, filename="model.bin")
    model = fasttext.load_model(model_path)

    # Collect texts for LID batch prediction
    source_texts = [example["translation"][NLLB_LANGUAGE_IDS[idx1]].lower().replace("ring", "") for example in cleaned_dataset] # TODO: make this more general
    target_texts = [example["translation"][NLLB_LANGUAGE_IDS[idx2]].lower().replace("ring", "") for example in cleaned_dataset]

    # Function to check if desired language is in top predictions
    def is_desired_language(predictions, desired_lang):
        for label, score in zip(predictions[0], predictions[1]):
            if label == f"__label__{desired_lang}":
                return score
        return 0.0

    # Function to predict LID in batches
    def batch_predict(texts, lang_id, batch_size):
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"LID prediction for {lang_id}:"):
            batch = texts[i:i + batch_size]
            batch_results = model.predict(batch, k=3)
            results.extend(zip(*batch_results))
        scores = [is_desired_language(result, lang_id) for result in results]
        num_zero_scores = scores.count(0.0)
        logging.info(f"No. unmatched LID scores: {num_zero_scores}")
        return scores

    # Predict LID for source and target texts in batches
    source_lid_scores = batch_predict(source_texts, NLLB_LANGUAGE_IDS[idx1], 1000)
    target_lid_scores = batch_predict(target_texts, NLLB_LANGUAGE_IDS[idx2], 1000)

    # Filter by the lid scores
    translations = []
    laser_scores = []
    for i, example in enumerate(cleaned_dataset):
        if source_lid_scores[i] > lid_score_threshold and target_lid_scores[i] > lid_score_threshold:
            translations.append(example["translation"])
            laser_scores.append(example["laser_score"])

    sorted_indices = np.argsort(laser_scores)
    translations = np.array(translations)[sorted_indices]
    translations = translations[-max_size:]
    logging.info(f"Filtering completed. {len(translations)}/{len(cleaned_dataset)} translations passed the thresholds.")

    # Clean using the model
    # cleaned_translation = cleaner.predict_batch(translations)
    cleaned_translation = translations

    # Save to output_dir
    with gzip.open(os.path.join(output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx1]}.gz"), 'wt', encoding="utf-8") as src_out:
        src_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx1]].replace('\n', '') for t in cleaned_translation]))
    with gzip.open(os.path.join(output_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.nllbsub.{OPUS_LANGUAGE_IDS[idx2]}.gz"), 'wt', encoding="utf-8") as tgt_out:
        tgt_out.write('\n'.join([t[NLLB_LANGUAGE_IDS[idx2]].replace('\n', '') for t in cleaned_translation]))

def process_filtered_files(idx1, idx2, output_dir, clean_output_dir):
    counts_dict = {}
    for (source, target) in [(OPUS_LANGUAGE_IDS[idx1], OPUS_LANGUAGE_IDS[idx2]), (OPUS_LANGUAGE_IDS[idx2], OPUS_LANGUAGE_IDS[idx1])]:
        direction_output_dir = os.path.join(clean_output_dir, f"{source}{target}")
        os.makedirs(direction_output_dir, exist_ok=True)

        for subset in ["train", "test", "valid"]:
            bitext_list = []

            with gzip.open(os.path.join(output_dir, f"{source}-{target}.{source}.{subset}.gz"), 'rt', encoding="utf-8") as src_file, gzip.open(os.path.join(output_dir, f"{source}-{target}.{target}.{subset}.gz"), 'rt', encoding="utf-8") as tgt_file:
                with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{source}"), 'w', encoding="utf-8") as src_outfile, open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.{target}"), 'w', encoding="utf-8") as tgt_outfile:
                    for src_text, tgt_text in zip(src_file, tgt_file):
                        bitext_list.append({
                            "translation": {
                                source: src_text.strip("\n"),
                                target: tgt_text.strip("\n")
                            }
                        })
                        src_outfile.write(src_text)
                        tgt_outfile.write(tgt_text)

            with open(os.path.join(direction_output_dir, f"{subset}.{source}-{target}.json"), 'w', encoding="utf-8") as j:
                json.dump(bitext_list, j)
        
        counts_dict[source+"-"+target] = len(bitext_list)
    
    return counts_dict

def main(args):
    global CACHE_MANAGER
    CACHE_MANAGER = SqliteCacheManager()

    download_nusax(overwrite=args.overwrite_cache)
    download_seed(overwrite=args.overwrite_cache)
    download_nusa_writes(overwrite=args.overwrite_cache)
    download_minangnlp(overwrite=args.overwrite_cache)
    download_indonesianmnt(overwrite=args.overwrite_cache)

    # cleaner = Cleaner(args.cleaner, n_gpus=args.n_gpus)
    cleaner = None
    opus_dir = os.path.join(args.output_dir, "opus")
    clean_output_dir = os.path.join(args.output_dir, "clean")
    os.makedirs(opus_dir, exist_ok=True)

    for i, (lang1, lang2) in enumerate(args.language_pairs):
        idx1 = OPUS_LANGUAGE_IDS.index(lang1)
        idx2 = OPUS_LANGUAGE_IDS.index(lang2)

        run_output_dir = os.path.join(opus_dir, f"{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}")
        make_dir_if_not_exists(run_output_dir)

        config_filepath = write_template(idx1, idx2, args.template_yaml, run_output_dir, args.configs_dir)
        create_alignment_files(idx1, idx2, run_output_dir)

        download_nllb(idx1, idx2, cleaner, args.lid_model, args.laser_threshold[i], args.lid_threshold, args.max_nllb_size, run_output_dir)

        # Execute and wait pipeline execution
        opus_command = ["opusfilter", config_filepath]
        if args.overwrite_opus:
            opus_command.append("--overwrite")
        subprocess.run(opus_command, env=os.environ.update({"PYTHONPATH": os.getcwd()}), check=True)
        
        counts_dict_new = process_filtered_files(idx1, idx2, run_output_dir, clean_output_dir)
        print(counts_dict_new)



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
        "--cleaner",
        type=str,
        required=True,
        help="Path to finetuned cleaner llama-3 model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--laser_threshold",
        type=float,
        nargs="+",
        # default=1.07,
        help="Laser score threshold"
    )
    

    # Not required args
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--lid_model",
        type=str,
        default="cis-lmu/glotlid",
        help="LID model ID in the Huggingface Hub"
    )
    parser.add_argument(
        "--lid_threshold",
        type=float,
        default=0.9,
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