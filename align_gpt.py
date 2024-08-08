import json
import os
import openai
import time
import hashlib
import anthropic
from concurrent.futures import ThreadPoolExecutor

ALIGN_PROMPT = """
You will receive two sentences: one in a source language and one in a target language. 

Your task is:
1. Determine if the two sentences are aligned, meaning they show some resemblance of similarity.
2. If they have the same meaning, clean and align the sentences by fixing syntax errors, removing noise (such as unnecessary phrases, punctuation or ambiguous numbers), and normalizing text (e.g., capitalization).

Format your response as follows:
1. On the first line, respond with "False" if the sentences are clearly are not aligned, and "True" otherwise.
2. If the first line is "True", provide the cleaned and aligned sentences on the second and third lines respectively.
3. Seperate each pair by a newline.

Here is an example to guide you:

Input:
Indonesian: Dengan harga yang bisa dibilang menengah, apa saja yang ditwarkannya?
Balinese: Suratan puniki nénten indik Kabupatén miwah kota ring Kepulauan Riau.

Indonesian: Bahasa daerah memiliki karakteristik yang unik contohnya bahasa bali dengan tingkatan – tingkatan berbahasa yang menjadikan bahasa bali semakin menarik.
Balinese: (32:2) Basa daerah madue "karakteristik" sane soleh sakadi basa bali sane wenten "tingkatan-tingkatan" mabasa sane ngeranayang ipun "menarik".

Output:
False

True
Indonesian: Bahasa daerah memiliki karakteristik yang unik contohnya bahasa bali dengan tingkatan-tingkatan berbahasa yang menjadikan bahasa bali semakin menarik.
Balinese: Basa daerah madue karakteristik sane soleh sakadi basa bali sane wenten tingkatan-tingkatan mabasa sane ngeranayang ipun menarik.

Run this task for each sentence pair:
"""

SYSTEM_PROMPT = "You are AlignGPT, an expert in aligning and cleaning parallel sentences in different languages."

# class AlignGPT:
#     def __init__(self, batch_size=32):
#         self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
#         self.batch_size = batch_size
#         self.rate_limit = 4000  # Requests per minute
#         self.executor = ThreadPoolExecutor(max_workers=10)

#     def prepare_prompt(self, parallel_sentences, source_language, target_language):
#         prompt = ALIGN_PROMPT + "\n"
#         for source, target in parallel_sentences:
#             prompt += f"{source_language}: {source}\n{target_language}: {target}\n\n"
#         return prompt

#     def send_request(self, prompt):
#         response = self.client.messages.create(
#             model="claude-3-5-sonnet-20240620",
#             max_tokens=4096,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response

#     def parallel_requests(self, prompts):
#         results = []
#         for prompt in prompts:
#             result = self.executor.submit(self.send_request, prompt)
#             results.append(result)
#             time.sleep(60 / self.rate_limit)  # Ensure we don't exceed the rate limit
#         return [res.result() for res in results]

#     def parse_results(self, results, source_language, target_language):
#         cleaned_sentences = []
#         valid_flags = []
        
#         for result in results:
#             response = result['response']['body']['choices'][0]["message"]["content"]
#             lines = [line for line in response.split('\n') if line != '']
            
#             i = 0
#             while i < len(lines):
#                 if lines[i].strip().lower() == 'true':
#                     valid_flags.append(True)
#                     source_cleaned = lines[i + 1].replace(f"{source_language}:", "").strip()
#                     target_cleaned = lines[i + 2].replace(f"{target_language}:", "").strip()
#                     cleaned_sentences.append((source_cleaned, target_cleaned))
#                     i += 3
#                 else:
#                     valid_flags.append(False)
#                     cleaned_sentences.append(("", ""))
#                     i += 1

#         return cleaned_sentences, valid_flags

#     def compute_hash(self, parallel_sentences, source_language, target_language):
#         hash_input = json.dumps(parallel_sentences) + source_language + target_language
#         return hashlib.md5(hash_input.encode()).hexdigest()

#     def run_batch_alignment(self, parallel_sentences, source_language, target_language):
#         cache_dir = ".cache/align-gpt/"
#         os.makedirs(cache_dir, exist_ok=True)
#         hash_value = self.compute_hash(parallel_sentences, source_language, target_language)
#         cache_path = os.path.join(cache_dir, f"{hash_value}.jsonl")

#         if os.path.exists(cache_path):
#             print(f"Using cached results from {cache_path}")
#             results = []
#             with open(cache_path, "r") as f:
#                 for line in f:
#                     results.append(json.loads(line))
#             cleaned_sentences, valid_flags = self.parse_results(results, source_language, target_language)
#             return cleaned_sentences, valid_flags
        
#         prompts = [self.prepare_prompt(parallel_sentences[i:i + self.batch_size], source_language, target_language) for i in range(0, len(parallel_sentences), self.batch_size)]
#         responses = self.parallel_requests(prompts)

#         with open(cache_path, "w") as f:
#             for response in responses:
#                 f.write(json.dumps(response) + "\n")

#         cleaned_sentences, valid_flags = self.parse_results(responses, source_language, target_language)
#         print(f"Batch completed. Results retrieved.")
#         return cleaned_sentences, valid_flags


class AlignGPT:
    def __init__(self, batch_size=32):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.batch_size = batch_size

    def prepare_batch_file(self, parallel_sentences, source_language, target_language, model="gpt-4o-mini"):
        batch_data = []
        for i in range(0, len(parallel_sentences), self.batch_size):
            batch_sentences = parallel_sentences[i:i+self.batch_size]
            prompt = ALIGN_PROMPT + "\n"
            for source, target in batch_sentences:
                prompt += f"{source_language}: {source}\n{target_language}: {target}\n\n"
            
            custom_id = f"request-{i // self.batch_size + 1}"
            batch_data.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    "max_tokens": 4096
                }
            })

        with open("batchinput.jsonl", "w") as f:
            for entry in batch_data:
                f.write(json.dumps(entry) + "\n")
        
        return "batchinput.jsonl"

    def upload_batch_file(self, file_path):
        response = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        return response.id

    def create_batch(self, file_id):
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "alignment job"
            }
        )
        return response.id

    def check_batch_status(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        return response.status

    def retrieve_batch_results(self, batch_id, cache_path):
        batch_details = self.client.batches.retrieve(batch_id)
        output_file_id = batch_details.output_file_id
        
        file_content = self.client.files.content(output_file_id).text

        with open(cache_path, "w") as f:
            f.write(file_content)
        
        results = []
        with open(cache_path, "r") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    def parse_results(self, results, source_language, target_language):
        cleaned_sentences = []
        valid_flags = []
        
        for result in results:
            response = result.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            lines = [line for line in response.split('\n') if line != '']
            
            i = 0
            try:
                while i < len(lines):
                    if lines[i].strip().lower() == 'true':
                        valid_flags.append(True)
                        source_cleaned = lines[i + 1].replace(source_language+":", "").strip()
                        target_cleaned = lines[i + 2].replace(target_language+":", "").strip()
                        cleaned_sentences.append((source_cleaned, target_cleaned))
                        i += 3
                    else:
                        valid_flags.append(False)
                        cleaned_sentences.append(("", ""))
                        i += 1
            except IndexError:
                print(result)

        return cleaned_sentences, valid_flags

    def compute_hash(self, parallel_sentences, source_language, target_language):
        hash_input = json.dumps(parallel_sentences) + source_language + target_language
        return hashlib.md5(hash_input.encode()).hexdigest()

    def run_batch_alignment(self, parallel_sentences, source_language, target_language):
        cache_dir = ".cache/align-gpt/"
        os.makedirs(cache_dir, exist_ok=True)
        hash_value = self.compute_hash(parallel_sentences, source_language, target_language)
        cache_path = os.path.join(cache_dir, f"{hash_value}.jsonl")

        if os.path.exists(cache_path):
            print(f"Using cached results from {cache_path}")
            results = []
            with open(cache_path, "r") as f:
                for line in f:
                    results.append(json.loads(line))
            cleaned_sentences, valid_flags = self.parse_results(results, source_language, target_language)
            return cleaned_sentences, valid_flags
        
        file_path = self.prepare_batch_file(parallel_sentences, source_language, target_language)
        file_id = self.upload_batch_file(file_path)
        batch_id = self.create_batch(file_id)
        
        print(f"Batch {batch_id} created. Waiting for completion...")

        while True:
            status = self.check_batch_status(batch_id)
            if status in ["completed", "failed", "expired"]:
                break
            print(f"Batch status: {status}. Checking again in 10 seconds...")
            time.sleep(10)

        if status == "completed":
            raw_results = self.retrieve_batch_results(batch_id, cache_path)
            cleaned_sentences, valid_flags = self.parse_results(raw_results, source_language, target_language)
            print(f"Batch {batch_id} completed. Results retrieved.")
            return cleaned_sentences, valid_flags
        else:
            print(f"Batch {batch_id} failed with status: {status}.")
            return None, None

# Example usage:
# parallel_sentences = [
#     ("With a price that can be considered moderate, what does it offer?!!", "Suratan puniki nénten indik Kabupatén miwah kota ring Kepulauan Riau."),
#     ("Regional languages have unique characteristics, for example, Balinese with levels of language that make Balinese increasingly interesting.", "(32:2) Basa daerah madue \"karakteristik\" sane soleh sakadi basa bali sane wenten \"tingkatan-tingkatan\" mabasa sane ngeranayang ipun \"menarik\"."),
#     ("Someone said, 'This is a good start'", "Tiang matur, 'Niki wiadin pemulaan sane becik'"),
#     ("Data must be normalized to be more consistent", "Data kantun dinormalisasi supaya langkung konsisten"),
#     ("(23:4) Before eating, make sure your hands are clean.", "(23:4) Sadurunge mangan, priksa tangan sampeyan resik."),
# ]

# source_language = "English"
# target_language = "Balinese"

# align_gpt = AlignGPT(batch_size=3)
# results, valid_flags = align_gpt.run_batch_alignment(parallel_sentences, source_language, target_language)

# print(results)
# print(valid_flags)