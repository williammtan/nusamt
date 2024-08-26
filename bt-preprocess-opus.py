import os
import subprocess
import json
import gzip

CACHE_DIR = os.path.join(os.getenv('CACHE_DIR', '.cache'), "wiki")

# ENVIRONMENT SETTINGS
PREPROCESS_CACHE_DIR = os.path.join( ".cache", "preprocess")


LANGUAGE_NAMES = {'ban': 'Balinese', 'ace': 'Acehnese', 'bjn': 'Banjar', 'bug': 'Buginese', 'min': 'Minangkabau', 'su': 'Sundanese', 'jv': 'Javanese', 'id': 'Indonesian', 'en': 'English'}
NLLB_LANGUAGE_NAMES = {'ban': 'ban_Latn', 'ace': 'ace_Latn', 'bjn': 'bjn_Latn', 'bug': 'bug_Latn', 'min': 'min_Latn', 'su': 'sun_Latn', 'jv': 'jav_Latn', 'id': 'ind_Latn', 'en': 'eng_Latn'}

def replace_multiple(string, dic):
    for k,v in dic.items():
        string = string.replace(k, v)
    return string


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


def main(args):
    clean_output_dir = os.path.join(args.output_dir, "clean")
    opus_dir = os.path.join(args.output_dir, "opus")

    for direction in args.language_directions.split(','):
        lang1, lang2 = direction.split('-')
        run_output_dir = os.path.join(opus_dir, f"{lang1}-{lang2}")

        config_filepath = write_template(lang1, lang2, args.template_yaml, run_output_dir, args.configs_dir)
        opus_command = ["opusfilter", config_filepath]
        if args.overwrite_opus:
            opus_command.append("--overwrite")
        subprocess.run(opus_command, env=os.environ.update({"PYTHONPATH": os.getcwd()}), check=True)

        process_filtered_files(lang1, lang2, run_output_dir, clean_output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and translate Wikipedia dumps.")
    parser.add_argument('--language_directions', required=True, help="Language direction pairs (source-target) combined with commas (e.g., en-ban,ban-en...)")
    parser.add_argument('--output_dir', required=True, help="Output directory to dump the generated translations")
    parser.add_argument("--template_yaml", type=str, default="configs/pipeline_bt_template.yaml", help="Path to the pipeline template YAML file")
    parser.add_argument("--configs_dir", type=str, default="configs/auto_pipeline/", help="Directory to store generated config files")
    parser.add_argument("--overwrite_opus", action="store_true", help="Overwrite opus files")

    args = parser.parse_args()

    main(args)