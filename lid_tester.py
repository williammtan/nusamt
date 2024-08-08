import argparse
import fasttext
from huggingface_hub import hf_hub_download

def download_model():
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    return model_path

def predict_language(model, text):
    predictions = model.predict(text, k=3)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict language using FastText model.")
    args = parser.parse_args()

    model_path = download_model()
    model = fasttext.load_model(model_path)

    print("Language prediction app. Type your text and press Enter to get predictions. Press Ctrl+C to exit.")

    try:
        while True:
            text = input("Enter text: ")
            predictions = predict_language(model, text)
            print(f"Predictions: {predictions}")
    except KeyboardInterrupt:
        print("\nExiting the app. Goodbye!")