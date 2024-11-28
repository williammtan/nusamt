import os
import gzip
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import openai
from spire.doc import Document, FileFormat
from pdf2docx import Converter

load_dotenv()
client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))


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

The last sentence pair provided:
<last_pair>
{LAST_PAIR}
</last_pair>

Follow these instructions to complete the task:

1. Identify and extract sentences that have translations in both the source and target language
2. For each pair of parallel sentences, use the following format:
     <parallel_sentences>
       <source_sentence>[Original sentence in its original language]</source_sentence>
       <target_sentence>[Corresponding sentence in the target language]</target_sentence>
     </parallel_sentences>
3. Apply some basic cleaning in the capitalization, formatting, and punctuation such that the source and target sentences are more aligned.
4. Additional guidelines:
   - If a sentence is incomplete or ambiguous, you need not include it
   - You can truncate "-" and join words that span across lines
   - Enclose your entire output within <parallel_sentences_list> tags
   - Do not say anything besides your output.

Begin processing the bilingual book content and generating parallel sentences according to these instructions.
"""

ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)

def parse_line(filepath):
    doc = DocumentFile.from_pdf(filepath.as_posix())
    result = ocr_model(doc)

    pages = []

    for page in result.pages:
        page_text = ""
        if len(page.blocks) != 1:
            continue
        
        for line in page.blocks[0].lines:
            for word in line.words:
                if word.confidence > 0.5:
                    page_text += word.value + " "
            page_text += "\n"
    
        pages.append(page_text.strip())

    return pages

def parse_html(filepath):
    cv = Converter(filepath.with_suffix('.pdf').as_posix())
    cv.convert(filepath.with_suffix('.docx').as_posix())
    cv.close()

    document = Document()
    document.LoadFromFile(filepath.with_suffix('.docx').as_posix())
    document.SaveToFile(filepath.with_suffix('.html').as_posix(), FileFormat.Html)
    document.Close()

    soup = BeautifulSoup(open(filepath.with_suffix('.html').as_posix()).read(), 'html.parser')

    for tag in soup.find_all(style=True):
        del tag['style']

    for tag in soup.find_all(class_=True):
        if not tag['class'][0].startswith('Section') or not tag['class'][0][7:].isdigit():
            del tag['class']

    def merge_spans(tag):
        for span in tag.find_all('span'):
            span.replace_with(span.text + "\n")

    merge_spans(soup)

    def trim_empty_elements(tag):
        for child in tag.contents[:]:
            if isinstance(child, str) and child.strip() == "":
                child.extract()
            elif child.name:
                trim_empty_elements(child)
                if len(child.contents) == 0:
                    child.extract()

    trim_empty_elements(soup)

    body_content = soup.body.find_all(recursive=False)
    pages = [str(element) for element in body_content]

    return pages

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

def clean_book(text, src, tgt, last_pair=""):
    prompt = PROMPT_TEMPLATE.format(BILINGUAL_BOOK=text, SOURCE_LANGUAGE=src, TARGET_LANGUAGE=tgt, LAST_PAIR=last_pair)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=16384
    )
    
    return response.choices[0].message.content

def main(args):
    df = pd.read_csv(args.csv_file)

    direction_map = {}

    for i, (_, row) in enumerate(df.iterrows()):
        print(f'Running Book #{i+1}: {row["file"]}')
        parse_func = PARSERS[row["type"]]
        filepath = Path(os.path.join(args.file_directory, row["directory"], row["file"]))

        raw_pages = parse_func(filepath)
        
        last_pair = ""
        for start_idx in range(0, len(raw_pages), args.batch_size):
            end_idx = min(start_idx + args.batch_size + args.window_size, len(raw_pages))
            batch_text = "\n".join([f"<page>{page}</page>" for page in raw_pages[start_idx:end_idx]])
            cleaned_html = clean_book(batch_text, src=LANGUAGE_NAMES_MAP[row["source"]], tgt=LANGUAGE_NAMES_MAP[row["target"]], last_pair=last_pair)
            last_pair = parse(cleaned_html)[-1] if cleaned_html else ""

            parallel_sentences = parse(cleaned_html)
            lang_pair = row["source"] + '-' + row["target"]
            if lang_pair not in direction_map:
                direction_map[lang_pair] = []
            
            direction_map[lang_pair].extend(parallel_sentences)
    
    os.makedirs(args.output, exist_ok=True)
    for direction in direction_map.keys():
        src, tgt = direction.split('-')
        with gzip.open(os.path.join(args.output, f'{src}-{tgt}.{src}.gz'), 'wb') as src_in, gzip.open(os.path.join(args.output, f'{src}-{tgt}.{tgt}.gz'), 'wb') as tgt_in:
            for src_sentence, tgt_sentence in direction_map[direction]:
                src_in.write((src_sentence + '\n').encode('utf-8'))
                tgt_in.write((tgt_sentence + '\n').encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("csv_file", help="Name of the CSV file to process")
    parser.add_argument("file_directory", help="Directory containing the CSV file")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for pages")
    parser.add_argument("--window_size", type=int, default=1, help="Number of pages before and after each batch to include")

    args = parser.parse_args()

    main(args)