
GIT vm setup
```bash
ssh-keygen -t ed25519 -C "william@tan.id"
cat ~/.ssh/id_ed25519.pub
```

Go to https://github.com/settings/keys and grant access to the public key

```bash
git remote set-url origin git@github.com:williammtan/ALMA.git
  git config --global user.email "william@tan.id"
  git config --global user.name "will tan"
```

VLLM TESTING
``` bash
python -m vllm.entrypoints.openai.api_server --model {model_path}
```


SETUP FOR mwoffliner
```bash
sudo apt install lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo add-apt-repository ppa:redislabs/redis
sudo apt-get update
sudo apt-get install redis

cd ~
curl -sL https://deb.nodesource.com/setup_16.x -o /tmp/nodesource_setup.sh

sudo bash /tmp/nodesource_setup.sh

sudo apt install nodejs npm

npm i -g mwoffliner

```

GCLOUD AUTH
https://cloud.google.com/compute/docs/authentication
``` bash
gcloud storage cp {src} {tgt}
```

MB experiment

```bash
python -c "from preprocess import get_lang_directions; print(','.join(['-'.join(d) for d in get_lang_directions(['ban', 'min'], ['id', 'en'], order_matters=False)]))"
```

VLLM Setup
```python

in /opt/conda/lib/python3.10/site-packages/vllm/transformers_utils/detokenizer.py

    if new_token_id >= 32000:
        new_text = " " + new_text
```

1. Preprocess DONE

```bash
python preprocess.py --language_pairs ban-id,id-min,ban-en,en-min --cleaner data/cleaner/model/ --output_dir data/mb/ --lid_threshold 0.9 --n_gpus 4

python preprocess.py --language_pairs ban-id,id-min,ban-en,en-min --cleaner data/cleaner/model/ --output_dir data/mb/ --lid_threshold 0.9 --laser_threshold 1.05 --n_gpus 2

python preprocess.py --language_pairs ban-id,ban-en --cleaner data/cleaner/model/ --output_dir data/ban-testing/ --lid_threshold 0.9 --laser_threshold 1.25 1.09 --n_gpus 2 --overwrite_opus
```

2. First training WORKING ON

```bash
OUTPUT_DIR=${1:-"./data/mb-final/train/"}
pairs=${2:-"ban-en,ban-id,en-ban,en-min,id-ban,id-min,min-en,min-id"}
LORA_RANK=${3:-"16"}
export HF_TOKEN="hf_tokenxyz"
export CXX=g++-11
export CC=gcc-11
export LD=g++-11

# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config.yaml \
     run_llmmt.py \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --torch_dtype "bfloat16" \
    --mmt_data_path  data/mb-final/clean/ \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --bf16 \
    --learning_rate 2e-3 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to none \
```

3. Eval first training

```bash
MODEL_DIR="./data/mb/train/"
OUTPUT_DIR="./data/mb/eval/"
TEST_PAIRS="ban-en,ban-id,en-ban,en-min,id-ban,id-min,min-en,min-id"
export HF_TOKEN="hf_tokenxyz"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_eval_config.yaml \
    run_llmmt.py \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --torch_dtype "bfloat16" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path data/flores-eval/ \
    --per_device_eval_batch_size 6 \
    --output_dir ${OUTPUT_DIR} \
    --use_peft \
    --peft_model_id ${MODEL_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir 


## Evaluation (BLEU, COMET)
bash ./evals/eval_generation_fix.sh ${OUTPUT_DIR} ${TEST_PAIRS} ./data/flores-eval
```

3. Backtranslation
(a) Find wiki texts (download from wiki, clean, select x sentences)

PROMPT:
Make a python script that given language directions (source language-target langauge), downloads and preprocesses wikipedia dumps into distinct sentences, and translate the corresponding sentences to all the directions, dumping to a output directory.

Inputs:
Required:
- Language directions: language direction pairs (source-target) eg. en-ban all together combined with commas (eg. en-ban,ban-en...)
- Output dir: output dir in which to dump the translation. The dir should contain all the language directions with the commas stripped (eg. enban, banen) and each of the lang direction dirs should contain test.{source language}-{target language}.json files containing a list of objects each in the format of:
{
    "translation": {
        source lang: "", # the sentence to be translated
        target lang: "" # an empty string
    }
}
Optional:
- Wiki dumps url: default = https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2 where {} indicates the substituted language code (eg. jv/ban/en)
- Max sentences per direction: default = 100000
- Min words: default 5, minimum number of words per sentence to be used.

Process:
1. Accept inputs, split the language directions into list of tuples for easier accessing
2. Iterating over all possibile languages in the directions given:
    (a) Check if the "wiki/{language}" subdirectory in the cache directory (cache directory should be configured as the CACHE_DIR environment variable or ".cache" by default) is already defined. If it is, skip to step (c). Else continue to step (b)
    (b) Download the wiki dump to a tempfile by the wiki dumps url substituting the language in the {}
    (c) Run "python -m wikiextractor.WikiExtractor {wiki dump file} -o {wiki/{language} cache dir}" on the tempfile
    (d) Extract the sentences by iterating over all the files (which could be in a subdirectory so use glob) in the wiki/{language} cache dir and run extract_and_split_sentences.
    (e) Create a list of all language directions that have the target language equal to the current language (saving the number of iterations as N)
    (f) Split the sentences into N partitions.
    (g) Iterate over this list and set each index of iteration as i:
        (i) Take the ith partition from the sentence partition
        (ii) Save all the languages to the file {output dir}/{source language}-{target language}/test.{source language}-{target language}.json in the format given above.


Supplimental code:

``` python
def extract_and_split_sentences(file_path):
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
            if len(sentence.split(' ')) > 5 and '&gt;' not in sentence and '&lt;' not in sentence:
                yield sentence.strip().replace('\n', ' ')
```

```bash
python bt-preprocess.py --language_directions "ban-en,ban-id,en-ban,id-ban" --output_dir "data/bt/examples" --model ./data/ban/train/ --url_json_path configs/wiki_urls.json
```

(b) Predict with model
```bash
OUTPUT_DIR="./data/bt/prediction/"
TEST_PAIRS="ban-en,ban-id,en-ban,en-min,id-ban,id-min,min-en,min-id"
DATA_DIR="./data/bt/input_prediction"
MODEL_DIR="./data/train/checkpoint-11664/"
export HF_TOKEN="hf_tokenxyz"
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_eval_config.yaml \
    run_llmmt.py \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --torch_dtype "bfloat16" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ${DATA_DIR} \
    --per_device_eval_batch_size 6 \
    --output_dir ${OUTPUT_DIR} \
    --use_peft \
    --peft_model_id ${MODEL_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir 
```

(c) Prepare bt-train dataset

a-c:
```bash
python bt-preprocess.py --language_directions "ban-en,ban-id,en-ban,id-ban" --model data/ban/train-model/ --output_dir "data/bt/examples/" --url_json_path configs/wiki_urls.json --max_sentences 200000
```

(d) Train (1 EPOCH!)
``` bash
OUTPUT_DIR="data/bt/train"
DATA_DIR="data/bt/examples"
pairs="ban-en,ban-id,en-ban,en-min,id-ban,id-min,min-en,min-id"
MODEL_PATH="data/mb/train/checkpoint-26620/"
LORA_RANK=${3:-"16"}
export CXX=g++-11
export CC=gcc-11
export LD=g++-11

# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config.yaml \
     run_llmmt.py \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --torch_dtype "bfloat16" \
    --mmt_data_path  ${DATA_DIR} \
    --use_peft \
    --peft_model_id ${MODEL_PATH} \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --bf16 \
    --learning_rate 2e-3 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to none \
```

run BT:

```bash
python merge_peft.py -m yellow-AI-NLP/komodo-7b-base -t yellow-AI-NLP/komodo-7b-base -p data/ban/train/checkpoint-25973/ -o data/ban/train-model/
```

```bash
python bt-preprocess.py --language_directions "ban-en,ban-id,en-ban,id-ban" --model data/ban/train-model/ --output_dir "data/bt/examples/" --url_json_path configs/wiki_urls.json --max_sentences 100000 && bash run-bt-train.sh
```


vllm serve function:
```bash
python -m vllm.entrypoints.openai.api_server --model data/ban/train-model --served-model-name nusa-7b-ban --tensor-parallel-size 2 --api-key "the_api_key__"
```


the plann:

# REPEAT change the thing to next one and keep doing this
cp -r data/ban/opus data/ban-align

python preprocess.py --language_pairs "ban-id,ban-en" --cleaner data/cleaner/model/ --output_dir data/ban-align/ --lid_threshold 0.9 --laser_threshold 1.25 1.09 --n_gpus 2 --overwrite_opus

# note the counts
zcat data/ban-align/opus/ban-en/ban-en.en.filtered.gz | wc -l
zcat data/ban-align/opus/ban-id/ban-id.id.filtered.gz | wc -l
zcat data/ban-align/opus/ban-en/ban-en.en.aligned.gz | wc -l
zcat data/ban-align/opus/ban-id/ban-id.id.aligned.gz | wc -l

# END REPEAT #

## TEMP MOVING ##

# bible verses
mv data/ban-align/opus data/ban-align/opus-bv

# baliwiki
mv data/ban-align/opus data/ban-align/opus-bw

# mined
mv data/ban-align/opus data/ban-align/opus-nl


# handwritten - REMOVE THE ALIGNMENTS (but remember to change to .aligned.gz after preprocessing)
mv data/ban-align/opus data/ban-align/opus-hw


# in the end
mkdir -p data/ban-align/opus/ban-id
mkdir -p data/ban-align/opus/ban-en
zcat data/ban-align/opus-nl/ban-en/ban-en.en.aligned.gz data/ban-align/opus-bw/ban-en/ban-en.en.aligned.gz data/ban-align/opus-bv/ban-en/ban-en.en.aligned.gz data/ban-align/opus-hw/ban-en/ban-en.en.aligned.gz | gzip > data/ban-align/opus/ban-en/ban-en.en.aligned.gz
zcat data/ban-align/opus-nl/ban-id/ban-id.id.aligned.gz data/ban-align/opus-bw/ban-id/ban-id.id.aligned.gz data/ban-align/opus-bv/ban-id/ban-id.id.aligned.gz data/ban-align/opus-hw/ban-id/ban-id.id.aligned.gz | gzip > data/ban-align/opus/ban-id/ban-id.id.aligned.gz
zcat data/ban-align/opus-nl/ban-en/ban-en.ban.aligned.gz data/ban-align/opus-bw/ban-en/ban-en.ban.aligned.gz data/ban-align/opus-bv/ban-en/ban-en.ban.aligned.gz data/ban-align/opus-hw/ban-en/ban-en.ban.aligned.gz | gzip > data/ban-align/opus/ban-en/ban-en.ban.aligned.gz
zcat data/ban-align/opus-nl/ban-id/ban-id.ban.aligned.gz data/ban-align/opus-bw/ban-id/ban-id.ban.aligned.gz data/ban-align/opus-bv/ban-id/ban-id.ban.aligned.gz data/ban-align/opus-hw/ban-id/ban-id.ban.aligned.gz | gzip > data/ban-align/opus/ban-id/ban-id.ban.aligned.gz



python bt-preprocess.py --language_directions "ban-en,ban-id,en-ban,id-ban" --model data/ban-align/train-model/ --output_dir "data/bt-align/examples/" --url_json_path configs/wiki_urls.json --max_sentences 100000


MIN

python preprocess.py --language_pairs "id-min,en-min" --cleaner data/cleaner/model/ --output_dir data/min/ --lid_threshold 0.9 --laser_threshold 1.09 1.09 --n_gpus 2 --overwrite_opus
