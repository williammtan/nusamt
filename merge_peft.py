import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def main(model_name, tokenizer_name, peft_path, output_path):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        trust_remote_code=True
    )

    # Load PEFT model
    model = PeftModel.from_pretrained(
        base_model, 
        peft_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload"
    )


    # Merge and unload the model
    model = model.merge_and_unload()

    # Save the model and tokenizer to the output path
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge and save PEFT model')
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Name or path of the Hugging Face model')
    parser.add_argument('--tokenizer_name', '-t', type=str, required=True, help='Name of the Hugging Face tokenizer')
    parser.add_argument('--peft_path', '-p', type=str, required=True, help='Path to the PEFT model')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Output path for the merged model')

    args = parser.parse_args()

    main(args.model_name, args.tokenizer_name, args.peft_path, args.output_path)