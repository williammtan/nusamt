import csv
import gzip
import os
import argparse
from tqdm import tqdm
import re

from sklearn.metrics.pairwise import cosine_similarity
from laser_encoders import LaserEncoderPipeline

# Initialize the LASER encoder pipelines for different languages
NLLB_LANGUAGE_NAMES = {
    'ban': 'ban_Latn', 'min': 'min_Latn', 'id': 'ind_Latn', 'en': 'eng_Latn'
}

laser_encoders = {lang: LaserEncoderPipeline(lang=NLLB_LANGUAGE_NAMES[lang]) for lang in NLLB_LANGUAGE_NAMES}


# Function to read the CSV file and load data into a dictionary
def load_csv(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lang = row['language']
            chapter = row['chapter']
            verse = row['verse_number']
            line = row['line_number']
            text = row['text']
            key = (chapter, verse, line)
            if key not in data:
                data[key] = {}
            data[key][lang] = text
    
    pattern = re.compile(r'^\(\d+:\d+\)$')

    for (chapter, verse, line), translations in data.items():
        for lang in list(translations):
            text = data[((chapter, verse, line))][lang]
            found_redirection = pattern.search(text)
            if found_redirection:
                # remove the current, and remove the redirection of the same language
                del data[(chapter, verse, line)][lang]
                refered_verse, refered_line = found_redirection.string.replace('(', '').replace(')', '').split(':')
                try:
                    del data[(chapter, refered_verse, refered_line)][lang]
                except KeyError:
                    pass
    return data

# Function to write sentences to gzip files
def write_to_gzip(output_dir, source_lang, target_lang, sentences):
    os.makedirs(f'{output_dir}/{source_lang}-{target_lang}', exist_ok=True)
    source_file = f'{output_dir}/{source_lang}-{target_lang}/{source_lang}-{target_lang}.{source_lang}.gz'
    target_file = f'{output_dir}/{source_lang}-{target_lang}/{source_lang}-{target_lang}.{target_lang}.gz'
    
    with gzip.open(source_file, 'wt', encoding='utf-8') as src, gzip.open(target_file, 'wt', encoding='utf-8') as tgt:
        for src_sentence, tgt_sentence in sentences:
            src.write(src_sentence + '\n')
            tgt.write(tgt_sentence + '\n')

def get_highest_scoring_pair(sentences1, sentences2, src_lang, tgt_lang):
    embeddings1 = laser_encoders[src_lang].encode_sentences(sentences1)
    embeddings2 = laser_encoders[tgt_lang].encode_sentences(sentences2)
    sim_matrix = cosine_similarity(embeddings1, embeddings2)
    
    best_score = -1
    best_pair = (None, None)
    
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            if sim_matrix[i][j] > best_score:
                best_score = sim_matrix[i][j]
                best_pair = (sentences1[i], sentences2[j])
    
    return best_pair

# Main function to process the data and generate parallel sentences
def process_parallel_sentences(csv_file, language_pairs, lang_map, output_dir):
    data = load_csv(csv_file)
    
    for pair in language_pairs:
        src, tgt = pair.split('-')
        src_langs = lang_map[src]
        tgt_langs = lang_map[tgt]

        sentence_pairs = []

        for _, translations in tqdm(data.items()):
            src_translations = [translations[i] for i in src_langs if i in translations.keys()]
            tgt_translations = [translations[i] for i in tgt_langs if i in translations.keys()]
            if len(src_translations) > 0 and len(tgt_translations) > 0:
                best_pair = get_highest_scoring_pair(src_translations, tgt_translations, src, tgt)
                if best_pair[0] and best_pair[1]:
                    sentence_pairs.append(best_pair)
                
                # Write the sentences to gzip files
        write_to_gzip(output_dir, src, tgt, sentence_pairs)

# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Process parallel Bible verses.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the parallel verses.')
    parser.add_argument('language_pairs', type=str, nargs='+', help='List of language pairs (e.g., ban-en ban-id).')
    parser.add_argument('--lang_map', type=str, help='Dictionary mapping of language codes (e.g., \'{"ban": "bali", "en": "net", "id": "ayt"}\').')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory to save the gzip files.')

    args = parser.parse_args()
    
    # Convert lang_map from string to dictionary
    import ast
    args.lang_map = ast.literal_eval(args.lang_map)
    
    return args

# Main execution
if __name__ == '__main__':
    args = parse_args()
    process_parallel_sentences(args.csv_file, args.language_pairs, args.lang_map, args.output_dir)


lang_map = {"ban": ["bali"],"id": ["ayt","tb","tl","milt","sb2010","ks2011","kskk","vmd","tsi","bis","tmv","fayh","ende","sbdr","ldkdr","avb"],"en": ["net","nasb","hcsb","leb","niv","esv","nrsv","reb","nkjv","av","amp","nlt","gnb","erv","bbe","msg","cev","cevuk","gwv"]}
