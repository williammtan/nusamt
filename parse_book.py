import os
import gzip
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from spire.doc import Document, FileFormat
from spire.doc.common import *
from pdf2docx import Converter

# import vertexai
# from vertexai.generative_models import GenerativeModel
# import vertexai.preview.generative_models as generative_models

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

LANGUAGE_NAMES_MAP = {"id": "Indonesian", "ban": "Balinese", "min": "Minangkabau"}

PROMPT_TEMPLATE = """
You are tasked with generating parallel sentences based on a bilingual book. The book content will be provided to you, along with a specified source and target language. Your goal is to extract sentences from the book and generate parallel sentences in the source and target language.

Here is the bilingual book content:
<bilingual_book>
{BILINGUAL_BOOK}
</bilingual_book>

The source language:
<source_language>{SOURCE_LANGUAGE}</source_language>
The target language:
<target_language>{TARGET_LANGUAGE}</target_language>

Follow these instructions to complete the task:

1. Identify and extract sentences that have translations in both the source and target language
2. For each pair of parallel sentences, use the following format:
     <parallel_sentences>
       <source_sentence>[Original sentence in its original language]</source_sentence>
       <target_sentence>[Corresponding sentence in the target language]</target_sentence>
     </parallel_sentences>
3. Apply some basic cleaning in the capitalization, formatting and punctuation such that the source and target sentences are more aligned.
4. Additional guidelines:
   - If a sentence is incomplete or ambiguous, you need not include it
   - You can truncate "-" and join words that span across lines
   - Enclose your entire output within <parallel_sentences_list> tags
   - Do not say anything besides your output.

Begin processing the bilingual book content and generating parallel sentences according to these instructions.
"""

CONTINUE_PROMPT = "Please continue generating parallel sentences from where you left off, without starting a new <parallel_sentences_list> tag. Continue directly from the last word."

GENERATION_CONFIG = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_REGION"))

gemini = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=GENERATION_CONFIG,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)

def parse_line(filepath):
    doc = DocumentFile.from_pdf(filepath.as_posix())
    result = ocr_model(doc)

    text = ""

    for page in result.pages:
        page_text = "<page>"
        if len(page.blocks) != 1:
            continue
        
        for line in page.blocks[0].lines:
            for word in line.words:
                if word.confidence > 0.5:
                    page_text += word.value + " "
            page_text += "\n"
    
        text += page_text + "</page>\n\n"

    return text

def parse_html(filepath):
    cv = Converter(filepath.with_suffix('.pdf').as_posix())
    cv.convert(filepath.with_suffix('.docx').as_posix())
    cv.close()

    document = Document()
    document.LoadFromFile(filepath.with_suffix('.docx').as_posix())
    document.SaveToFile(filepath.with_suffix('.html').as_posix(), FileFormat.Html)
    document.Close()

    soup = BeautifulSoup(open(filepath.with_suffix('.html').as_posix()).read(), 'html.parser')

    # 1. Remove all style attributes
    for tag in soup.find_all(style=True):
        del tag['style']

    # 2. Remove classes that don't start with "Section" followed by a number
    for tag in soup.find_all(class_=True):
        if not tag['class'][0].startswith('Section') or not tag['class'][0][7:].isdigit():
            del tag['class']

    # 3. Merge span elements and replace with newlines
    def merge_spans(tag):
        for span in tag.find_all('span'):
            span.replace_with(span.text + "\n")

    merge_spans(soup)

    def trim_empty_elements(tag):
        for child in tag.contents[:]: # Iterate over a copy to allow safe removal
            if isinstance(child, str) and child.strip() == "":
                child.extract()
            elif child.name:  # Check if it's a tag
                trim_empty_elements(child)
                if len(child.contents) == 0: # If tag becomes empty after recursion
                    child.extract()

    trim_empty_elements(soup)

    # Print the cleaned HTML 
    return soup.prettify()


PARSERS = {
    "line": parse_line,
    "html": parse_html
}


def parse(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    parallel_sentences_html = soup.find_all('parallel_sentences')

    parallel_sentences = []

    for sentence_pair in parallel_sentences_html:
        source = sentence_pair.find('source_sentence')
        target = sentence_pair.find('target_sentence')
        if source and target:
            parallel_sentences.append([source.text, target.text])

    return parallel_sentences

def clean_book(text, src, tgt, max_continue=15):
    def send_message(prompt):
        # res = chat.send_message(
        #     prompt, 
        #     stream=True,
        # )
        # text_response = []
        # for chunk in tqdm(res):
        #     text_response.append(chunk.text)
        # return "".join(text_response)

        max_retries = 5
        retry_delay = 60  # in seconds

        for attempt in range(max_retries):
            try:
                res = chat.send_message(
                    prompt, 
                    stream=True,
                )
                text_response = []
                for chunk in tqdm(res):
                    text_response.append(chunk.text)
                return "".join(text_response)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Attempt {attempt + 1} failed: {e}. No more retries left.")
                    return "</parallel_sentences_list>" # force end
    
    init_prompt = PROMPT_TEMPLATE.format(BILINGUAL_BOOK=text, SOURCE_LANGUAGE=src, TARGET_LANGUAGE=tgt)

    chat = gemini.start_chat()
    raw_text = send_message(init_prompt)

    n_continue = 0
    while n_continue < max_continue:
        if raw_text.endswith("</parallel_sentences_list>"):
            break

        raw_text += send_message(CONTINUE_PROMPT)
        n_continue += 1

    return raw_text

def main(args):

    df = pd.read_csv(args.csv_file)

    direction_map = {}

    for i, (_, row) in enumerate(df.iterrows()):
        print(f'Running Book #{i+1}: {row["file"]}')
        parse_func = PARSERS[row["type"]]
        filepath = Path(os.path.join(args.file_directory, row["directory"], row["file"]))

        raw_text = parse_func(filepath)
        cleaned_html = clean_book(raw_text, src=LANGUAGE_NAMES_MAP[row["source"]], tgt=LANGUAGE_NAMES_MAP[row["target"]])
        parallel_sentences = parse(cleaned_html)

        lang_pair = row["source"]+'-'+row["target"]
        if lang_pair not in direction_map:
            direction_map[lang_pair] = []
        
        direction_map[lang_pair].extend(parallel_sentences)
    
    for direction in direction_map.keys():
        src, tgt = direction.split('-')
        with gzip.open(os.path.join(args.output, f'{src}-{tgt}.{src}.gz'), 'wb') as src_in, gzip.open(os.path.join(args.output, f'{src}-{tgt}.{tgt}.gz'), 'wb') as tgt_in:
            for src_sentence, tgt_sentence in direction_map[direction]:
                src_in.write(src_sentence + '\n')
                tgt_in.write(tgt_sentence + '\n')
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("csv_file", help="Name of the CSV file to process")
    parser.add_argument("file_directory", help="Directory containing the CSV file")
    parser.add_argument("--output", "-o", help="Output directory")
    
    args = parser.parse_args()

    main(args)

