set -e

OUTPUT_DIR=${1:-"./data/banmin/train/"}
pairs=${2:-"ban-en,ban-id,en-ban,id-ban,min-en,min-id,en-min,id-min"}
LORA_RANK=${3:-"16"}
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
    --mmt_data_path  data/banmin/clean/ \
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
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 2 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to wandb \

bash flores-eval.sh ./data/banmin/

# MODEL_DIR="./data/banmin/train/"
# OUTPUT_DIR="./data/banmin/eval/"
# TEST_PAIRS="ban-en,ban-id,en-ban,id-ban"
# export HF_TOKEN="hf_tokenxyz"
# # random port between 30000 and 50000
# port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

# accelerate launch --main_process_port ${port} --config_file configs/deepspeed_eval_config.yaml \
#     run_llmmt.py \
#     --model_name_or_path yellow-AI-NLP/komodo-7b-base \
#     --torch_dtype "bfloat16" \
#     --do_predict \
#     --low_cpu_mem_usage \
#     --language_pairs ${TEST_PAIRS} \
#     --mmt_data_path data/flores-eval-nld/ \
#     --per_device_eval_batch_size 6 \
#     --output_dir ${OUTPUT_DIR} \
#     --use_peft \
#     --peft_model_id ${MODEL_DIR} \
#     --predict_with_generate \
#     --max_new_tokens 256 \
#     --max_source_length 256 \
#     --bf16 \
#     --seed 42 \
#     --num_beams 5 \
#     --overwrite_cache \
#     --overwrite_output_dir 


## Evaluation (BLEU, COMET)
# bash ./evals/eval_generation_fix.sh ${OUTPUT_DIR} ${TEST_PAIRS} ./data/flores-eval-nld

# python merge_peft.py -m yellow-AI-NLP/komodo-7b-base -t yellow-AI-NLP/komodo-7b-base -p data/banmin/train/ -o data/banmin/train-model

# python -m vllm.entrypoints.openai.api_server --model data/banmin/train-model --served-model-name nusa-7b-ban --tensor-parallel-size 2 --api-key "6775a246-e9c1-45d4-80f5-bf2af51c2662"