OUTPUT_DIR=${1:-"./alma-7b-parallel-ft"}
pairs=${2:-"bug-en,bjn-en,ace-en,ban-en,en-bug,en-bjn,en-ace,en-ban,bjn-en,ace-en,ban-en,en-bug,en-bjn,en-ace,en-ban"} # ,bjn-en,ace-en,ban-en,en-bug,en-bjn,en-ace,en-ban
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export HF_TOKEN="hf_tokenxyz"
export TRANSFORMERS_CACHE=".cache/models/"

# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
accelerate launch --main_process_port ${port} --config_file configs/config_train_zero3.yaml \
     run_llmmt.py \
    --torch_dtype "float16" \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --mmt_data_path ./human_written_data/ \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --bf16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --eval_steps 0.1 \
    --save_strategy steps \
    --save_steps 0.1 \
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
    --overwrite_cache 

## Evaluation (BLEU, COMET)
bash ./evals/eval_generation.sh ${OUTPUT_DIR} ${pairs}