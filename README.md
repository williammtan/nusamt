# NusaMT-7B: Machine Translation for Low-Resource Indonesian Languages

This repository contains code and resources for reproducing the results presented in the paper "NusaMT-7B: Machine Translation for Low-Resource Indonesian Languages with Large Language Models." The repository includes instructions for setting up the environment, running evaluations, and testing NusaMT-7B against various models.

## Results


### Table 2: spBLEU Score Comparison of the LLaMA2-7B SFT Model with Various Enhancements

| Models                        | ban → en | en → ban | ban → id | id → ban |
|-------------------------------|----------|----------|----------|----------|
| LLaMA2-7B SFT                 | 27.63    | 13.94    | 27.90    | 13.68    |
| + Monolingual Pre-training    | 31.28    | 18.92    | 28.75    | 20.11    |
| + Mono + Backtranslation      | 33.97    | 20.27    | 29.62    | 20.67    |
| + Mono + LLM Cleaner          | 33.23    | 19.75    | 29.02    | 21.16    |
| + Mono + Cleaner + Backtrans. | **35.42**| **22.15**| **31.56**| **22.95**|

This table presents spBLEU scores for various configurations of the LLaMA2-7B model, showing the impact of monolingual pre-training, backtranslation, and LLM cleaning on translation performance across different language pairs.

### Table 3: spBLEU Scores of NusaMT-7B Compared Against SoTA Models and Large GPT Models

| Models                        | ban → en | en → ban | ban → id | id → ban | min → en | en → min | min → id | id → min |
|-------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| GPT-3.5-turbo, zero-shot      | 27.17    | 11.63    | 28.17    | 13.14    | 28.75    | 11.07    | 31.06    | 11.05    |
| GPT-4o, zero-shot             | 27.11    | 11.45    | 27.89    | 13.08    | 28.63    | 11.00    | 31.27    | 11.00    |
| GPT-4, zero-shot              | 27.20    | 11.59    | 28.41    | 13.24    | 28.51    | 10.99    | 31.00    | 10.93    |
| NLLB-600M                     | 33.96    | 16.86    | 30.12    | 15.15    | 35.05    | 19.72    | 31.92    | 17.72    |
| NLLB-1.3B                     | 37.24    | 17.73    | 32.42    | 16.21    | 38.59    | 22.79    | 34.68    | 20.89    |
| NLLB-3.3B                     | **38.57**| 17.09    | **33.35**| 14.85    | **40.61**| **24.71**| **35.20**| 22.44    |
| NusaMT-7B (Ours)              | 35.42    | **22.15**| 31.56    | **22.95**| 37.23    | 24.32    | 34.29    | **23.27**|

This table compares the performance of NusaMT-7B with state-of-the-art models and large GPT models in terms of spBLEU scores across multiple language pairs. NusaMT-7B shows significant improvements, particularly in translations into low-resource languages.


## Installation

To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Before running the experiments, prepare the FLORES-200 multilingual dataset by executing:

```bash
python make_flores.py <save_dir>
```

Replace `<save_dir>` with the directory where you want to save the FLORES-200 dataset.

## Reproducing Results

### Testing NusaMT-7B

To test NusaMT-7B, first download the model work directory:

```bash
git clone https://huggingface.co/xxx/xxx <your_work_directory>
```

Replace `<your_work_directory>` with the desired path where the model and evaluation files will be stored.

Then, run the evaluation script:

```bash
bash flores-eval.sh <work_directory> <pairs>
```

- `<work_directory>`: Directory where `/eval` is found and `/train` for the model weights.
- `<pairs>`: Comma-separated list of language pairs, e.g., `ban-en,ban-id,en-ban,id-ban`.

### Testing NLLB

You can test NLLB models using the provided Python script. Run the script using the following command:

```bash
python run_nllb.py --input_dir <input_dir> --output_dir <output_dir> --lang_pairs <lang_pairs> --batch_size <batch_size>
```

- `<input_dir>`: Directory containing the test files.
- `<output_dir>`: Directory where the predictions will be saved.
- `<lang_pairs>`: Comma-separated list of language pairs (e.g., `en-ban,ban-en`).
- `<batch_size>`: Batch size for NLLB inference (default is 64).

### Testing Meta

To test Meta models, run the following command:

```bash
bash run-meta-train.sh <output_dir> <pairs>
```

- `<output_dir>`: Directory where the output will be saved.
- `<pairs>`: Comma-separated list of language pairs, e.g., `ban-en,ban-id,en-ban,id-ban`.
