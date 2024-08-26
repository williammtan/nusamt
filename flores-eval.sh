WORKDIR=${1:-"./data/ban/"}
pairs=${2:-"ban-en,ban-id,en-ban,id-ban"}
LORA_RANK=${3:-"16"}
FLORES_DIR="./data/flores-200/"
OUTPUT_DIR="$WORKDIR/eval"
MODEL_DIR="$WORKDIR/train/"
export HF_TOKEN="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"
export HF_HOME=".cache/"
export CXX=g++-11
export CC=gcc-11
export LD=g++-11
export NCCL_P2P_LEVEL=NVL

# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config.yaml \
     run_llmmt.py \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --torch_dtype "bfloat16" \
    --mmt_data_path  ${FLORES_DIR} \
    --use_peft \
    --peft_model_id ${MODEL_DIR} \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
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
    --report_to wandb \


bash ./evals/eval_generation_fix.sh ${OUTPUT_DIR} ${pairs} ${FLORES_DIR}