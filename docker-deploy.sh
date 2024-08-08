docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model williamhtan/nusa-7b-bali