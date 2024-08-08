import os
import re
import subprocess
import tempfile
import json
import glob
import random
import gzip

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.model_selection import train_test_split

CACHE_DIR = os.path.join(os.getenv('CACHE_DIR', '.cache'), "wiki")

# ENVIRONMENT SETTINGS
PREPROCESS_CACHE_DIR = os.path.join( ".cache", "preprocess")

PROMPT_FORMAT = "Translate this from {lang1} to {lang2}:\n{lang1}: {sentence}\n{lang2}:"
LANGUAGE_NAMES = {'ban': 'Balinese', 'ace': 'Acehnese', 'bjn': 'Banjar', 'bug': 'Buginese', 'min': 'Minangkabau', 'su': 'Sundanese', 'jv': 'Javanese', 'id': 'Indonesian', 'en': 'English'}
NLLB_LANGUAGE_NAMES = {'ban': 'ban_Latn', 'ace': 'ace_Latn', 'bjn': 'bjn_Latn', 'bug': 'bug_Latn', 'min': 'min_Latn', 'su': 'sun_Latn', 'jv': 'jav_Latn', 'id': 'ind_Latn', 'en': 'eng_Latn'}

def replace_multiple(string, dic):
    for k,v in dic.items():
        string = string.replace(k, v)
    return string


class Translator:
    def __init__(self, model_path, temperature=0, n_gpus=2):
        self.model = LLM(model=model_path, tensor_parallel_size=n_gpus)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(max_tokens=256, temperature=temperature)
    
    def _format_prompt(self, sentence, lang1, lang2):
        lang1_name = LANGUAGE_NAMES[lang1]
        lang2_name = LANGUAGE_NAMES[lang2]
        return PROMPT_FORMAT.format(lang1=lang1_name, lang2=lang2_name, sentence=sentence)

    def predict_batch(self, sentences, lang1, lang2):
        """Cleans a list of translation dicts and returns a list of cleaned translation dicts."""

        formated_prompts = []

        for sentence in sentences:
            formated_prompts.append(self._format_prompt(sentence, lang1, lang2))
        
        results = self.model.generate(formated_prompts, self.sampling_params)
        cleaned_responses = [
            output.outputs[0].text
            for output in results
        ]
        return cleaned_responses


def save_translations(output_dir, lang_pair, sentences, partition):
    src, tgt = lang_pair.split('-')
    lang_dir = os.path.join(output_dir, f"{src}{tgt}")
    os.makedirs(lang_dir, exist_ok=True)
    output_file = os.path.join(lang_dir, f"{partition}.{src}-{tgt}.json")

    data = [{"translation": {src: sentence[0], tgt: sentence[1]}} for sentence in sentences]
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def extract_and_split_sentences(file_path, min_words=5):
    # Read the entire file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Regular expression to match each document
    doc_pattern = re.compile(r'<doc id="\d+" url="[^"]+" title="[^"]+">([\s\S]*?)</doc>')

    # Find all documents in the text
    documents = doc_pattern.findall(text)

    # Function to split text into sentences
    def split_into_sentences(text):
        # Here, we use a simple regex to split by sentence-ending punctuation followed by a space or end of string
        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        return sentence_pattern.split(text)

    # Process each document
    for doc in documents:
        # Remove the title (first line) and everything before the first \n\n
        doc_content = re.split(r'\n\n', doc, 1)[-1].replace('\n\n', ' ').strip()

        sentences = split_into_sentences(doc_content)
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out sentences that contain &gt; or &lt;
            if len(sentence.split(' ')) >= min_words and '&gt;' not in sentence and '&lt;' not in sentence:
                yield sentence.strip().replace('\n', ' ')

def download_wiki_dump(language, url):
    wiki_url = url.format(language, language)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bz2')
    tmp_file.close()
    subprocess.run(["wget", "-O", tmp_file.name, wiki_url], check=True)
    return tmp_file.name

def process_wiki_dump(language, dump_file, cache_dir):
    cache_path = os.path.join(cache_dir, f"{language}")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        subprocess.run(["python", "-m", "wikiextractor.WikiExtractor", dump_file, "-o", cache_path])
    return cache_path

def load_glotlib(language):
    return [ex['text'] for ex in load_dataset("cis-lmu/Glot500", name=language, split="train")]

def remove_duplicates(sentences):
    unique_sentences = list(set(sentences))

    print(f"Removed {len(sentences)-len(unique_sentences)} duplicate sentences.")
    return unique_sentences

def preprocess(language_directions, url_json_path, max_sentences=100000, min_words=5):
    language_pairs = [tuple(ld.split('-')) for ld in language_directions.split(',')]
    languages = set([lang for pair in language_pairs for lang in pair])

    with open(url_json_path, 'r', encoding='utf-8') as f:
        url_dict = json.load(f)

    directions_dict = {lang1+'-'+lang2: [] for lang1,lang2 in language_pairs}

    for language in languages:
        lang_sentences = []
        if language not in ['id', 'en']:
            lang_sentences.extend(load_glotlib(NLLB_LANGUAGE_NAMES[language]))

        cache_path = os.path.join(CACHE_DIR, f"{language}")
        if not os.path.exists(cache_path):
            dump_file = download_wiki_dump(language, url_dict[language])
            cache_path = process_wiki_dump(language, dump_file, CACHE_DIR)

        for file in glob.glob(os.path.join(cache_path, '**', '*'), recursive=True):
            if os.path.isfile(file):
                sentences = list(extract_and_split_sentences(file, min_words))
                lang_sentences.extend(sentences)


        random.shuffle(lang_sentences)

        num_partitions = len([pair for pair in language_pairs if pair[1] == language])
        lang_sentences = remove_duplicates(lang_sentences)[:max_sentences*num_partitions]
        partitions = [lang_sentences[i::num_partitions] for i in range(num_partitions)]

        idx = 0
        for src, tgt in language_pairs:
            if tgt == language:
                directions_dict[f"{src}-{tgt}"] = partitions[idx]
                idx+=1

        if len(lang_sentences) != 0:
            avg_word_count = sum(len(sentence.split()) for sentence in lang_sentences) / len(lang_sentences)
        else:
            avg_word_count = 0
        print(f"------------------{language}------------------")
        print(f"Number of sentences: {len(lang_sentences)}")
        print(f"Average word count: {avg_word_count:.2f}")
        print(f"Number of sentences per direction: {int(len(lang_sentences) / num_partitions)}")

    return directions_dict
    # # Print conclusion
    # for src, tgt in language_pairs:
    #     src_sentences = sentences_dict[src]
    #     avg_word_count = sum(len(sentence.split()) for sentence in src_sentences) / len(src_sentences)
    #     print(f"------------------{src}-{tgt}------------------")
    #     print(f"Number of sentences: {len(src_sentences)}")
    #     print(f"Average word count: {avg_word_count:.2f}")

def train_valid_test_split(sentences, valid_size, test_size):
    train, test_valid = train_test_split(sentences, test_size=valid_size+test_size)
    valid, test = train_test_split(test_valid, test_size=(test_size)/(valid_size+test_size))
    return train, valid, test

def write_template(src, tgt, template_yaml_path, output_dir, configs_dir): # alr /opus and dir created
    if not os.path.isdir(configs_dir):
        os.makedirs(configs_dir, exist_ok=True)

    with open(template_yaml_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    template = replace_multiple(template, {
        "{{source}}": src,
        "{{target}}": tgt,
        "{{source_fullname}}": LANGUAGE_NAMES[src],
        "{{target_fullname}}": LANGUAGE_NAMES[tgt],
        "{{nllb_source}}": NLLB_LANGUAGE_NAMES[src],
        "{{nllb_target}}": NLLB_LANGUAGE_NAMES[tgt],
        "{{output_directory}}": output_dir,
        "{{cache_dir}}": os.path.abspath(PREPROCESS_CACHE_DIR)
    })

    config_filepath = os.path.join(configs_dir, f"{src}-{tgt}_pipeline_bt.yaml")
    with open(config_filepath, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return config_filepath

def process_filtered_files(src, tgt, output_dir, clean_output_dir):
    for (source, target) in [(src, tgt), (src, tgt)]:
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

def main(language_directions, output_dir, model_path, url_json_path, max_sentences, min_words, template_yaml, configs_dir):
    model = Translator(model_path)
    directions_dict = preprocess(language_directions, url_json_path, max_sentences, min_words)

    opus_dir = os.path.join(output_dir, "opus")
    clean_output_dir = os.path.join(output_dir, "clean")

    for direction, sentences in directions_dict.items():
        lang1, lang2 = direction.split('-')
        translated_sentences = model.predict_batch(sentences, lang2, lang1)

        print([
            [translated_sentences[i], sentences[i]]
            for i in range(len(sentences))
        ][:10])

        run_output_dir = os.path.join(opus_dir, f"{lang1}-{lang2}")
        os.makedirs(run_output_dir, exist_ok=True)
        with gzip.open(os.path.join(run_output_dir, f"{lang1}-{lang2}.{lang1}.gz"), 'wt') as f:
            f.write('\n'.join([s.encode('utf-8') for s in translated_sentences]))
        
        with gzip.open(os.path.join(run_output_dir, f"{lang1}-{lang2}.{lang2}.gz"), 'wt') as f:
            f.write('\n'.join(sentences))

        config_filepath = write_template(lang1, lang2, template_yaml, run_output_dir, configs_dir)
        opus_command = ["opusfilter", config_filepath]
        if args.overwrite_opus:
            opus_command.append("--overwrite")
        subprocess.run(opus_command, env=os.environ.update({"PYTHONPATH": os.getcwd()}), check=True)

        process_filtered_files(lang1, lang2, run_output_dir, clean_output_dir)


        # train, valid, test = train_valid_test_split([
        #     [translated_sentences[i], sentences[i]]
        #     for i in range(len(sentences))
        # ], valid_size, test_size)

        # save_translations(output_dir, direction, train, "train")
        # save_translations(output_dir, direction, valid, "valid")
        # save_translations(output_dir, direction, test, "test")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and translate Wikipedia dumps.")
    parser.add_argument('--language_directions', required=True, help="Language direction pairs (source-target) combined with commas (e.g., en-ban,ban-en...)")
    parser.add_argument('--output_dir', required=True, help="Output directory to dump the generated translations")
    parser.add_argument('--model', required=True, help="Path to model")
    parser.add_argument('--url_json_path', required=True, help="Path to the JSON file containing the URLs for downloading the wikis")
    parser.add_argument('--max_sentences', type=int, default=100000, help="Maximum number of sentences per direction")
    parser.add_argument('--min_words', type=int, default=5, help="Minimum number of words per sentence")
    parser.add_argument("--template_yaml", type=str, default="configs/pipeline_bt_template.yaml", help="Path to the pipeline template YAML file")
    parser.add_argument("--configs_dir", type=str, default="configs/auto_pipeline/", help="Directory to store generated config files")

    args = parser.parse_args()

    main(args.language_directions, args.output_dir, args.model, args.url_json_path, args.max_sentences, args.min_words, args.template_yaml, args.configs_dir)