MODEL_DIR=$(find ./data/bt/train -type d -name 'checkpoint-*' | head -n 1)
OUTPUT_DIR="./data/bt/eval/"
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