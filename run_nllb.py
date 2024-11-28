import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HF_MODEL_NAME = "facebook/nllb-200-distilled-600M"

NLLB_DICT = {'ban': 'ban_Latn', 'ace': 'ace_Latn', 'bjn': 'bjn_Latn', 'bug': 'bug_Latn', 'min': 'min_Latn', 'su': 'sun_Latn', 'jv': 'jav_Latn', 'id': 'ind_Latn', 'en': 'eng_Latn'}

def chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def eval(model, tokenizer, ds, source_lang, target_lang, batch_size):
    inputs = []
    outputs_pred = []
    for example in ds:
        inputs.append(example["translation"][source_lang])

    for iss in tqdm(chunk(list(range(len(inputs))), batch_size), total=len(inputs) / batch_size):
        inputs_batch = [inputs[i] for i in iss]
        inputs_encoded = tokenizer(inputs_batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to("cuda:0")

        translated_tokens = model.generate(
            **inputs_encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_DICT[target_lang]), max_length=128
        )
        outputs_pred_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        outputs_pred.extend(outputs_pred_batch)

    return outputs_pred

def run_nllb(input_dir, output_dir, lang_pairs, batch_size):
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME).to("cuda:0")
    for pair in lang_pairs.split(','):
        source_lang, target_lang = pair.split('-')
        subdir = source_lang + target_lang
        test_file_path = os.path.join(input_dir, subdir, f'test.{source_lang}-{target_lang}.json')

        if not os.path.exists(test_file_path):
            print(f"File {test_file_path} not found.")
            continue

        with open(test_file_path, 'r') as f:
            test_data = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, src_lang=NLLB_DICT[source_lang])

        outputs_pred = eval(model, tokenizer, test_data, source_lang, target_lang, batch_size)

        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f'test-{source_lang}-{target_lang}')
        with open(output_file_path, 'w') as f:
            for sentence in outputs_pred:
                f.write(sentence + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run NLLB prediction on a directory')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the test files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the predictions')
    parser.add_argument('--lang_pairs', type=str, required=True, help='Comma-separated list of language pairs (e.g., en-ban,ban-en)')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size for NLLB inference')
    args = parser.parse_args()

    run_nllb(args.input_dir, args.output_dir, args.lang_pairs, args.batch_size)