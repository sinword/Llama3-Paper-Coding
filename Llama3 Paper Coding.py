import os
from openai import OpenAI
import fitz
import logging
import time
import re

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load the PDF file
pdf_file_path = "./Papers/Test_page_5.pdf"

prompt_v1 = """
Is there any sentence related to hardware? If yes, please list all the sentences directly without any additional comments or summaries.
Here are some examples:
* Virtual Reality (VR) and Augmented Reality (AR) Head Mounted Displays (HMDs) have the potential to significantly expand the display space, enabling immersive entertainment and workspaces that go beyond the physical limitations of the car interior.
* Problematically, VR HMDs also occlude visual perception of reality [44, 6] and thus the car’s motion, and are likely to lead to sensory mismatch and, consequently, motion sickness.
* However, assuming the orientation and velocity of the vehicle can be tracked at low latency, HMDs have the potential to portray the vehicle motion virtually.
Please only output sentences directly related to hardware. Do not include any additional comments or summaries. If the sentence is not related to hardware, do not output anything.
"""

prompt_v2 = """
Is there any sentence related to hardware? If yes, please list all the sentences directly without any additional comments or summaries.
Just print out the original sentence directly, do not reply anything else.
"""

prompt = prompt_v2

# Read the PDF file and parse the text
def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        with fitz.open(pdf_file_path) as pdf_file:
            for page_num in range(pdf_file.page_count):
                page = pdf_file.load_page(page_num)
                page_text = page.get_text().replace("\n", " ")
                text += page_text
        return text
    except Exception as e:
        logging.error(f"Error while extracting text from PDF: {e}")
        return None

# Split content into sentences
def split_into_sentences(text):
    sentences = re.split(r"(?<!\d)\.\s|\.\s(?!\d)", text)
    return sentences

def apply_model_to_sentence(sentence):
    try:
        completion = client.chat.completions.create(
            model="QuantFactory/Meta-Llama-3-8B-GGUF",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sentence}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error while applying model to sentence: {e}")
        return None

def main():
    start_time = time.time()  # Start timing
    
    text = extract_text_from_pdf(pdf_file_path)

    if text is None:
        return
    
    sentences = split_into_sentences(text)

    with open("output.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            processed_sentence = apply_model_to_sentence(sentence)
            if processed_sentence and processed_sentence.strip().lower() != "no":
                f.write(processed_sentence + "\n\n")
                print(processed_sentence)
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
