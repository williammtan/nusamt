export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export HF_TOKEN="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"
export TRANSFORMERS_CACHE=".cache/models/"


accelerate launch --config_file configs/deepspeed_eval_config.yaml \
    run_llmmt.py \
    --torch_dtype "float16" \
    --model_name_or_path Yellow-AI-NLP/komodo-7b-base \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-cs,cs-en \
    --mmt_data_path ./human_written_data_old/ \
    --per_device_eval_batch_size 1 \
    --output_dir ./output/ \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir
