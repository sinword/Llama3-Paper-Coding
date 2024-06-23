import os
import openai
from openai import OpenAI
import fitz
import logging
import time

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load the PDF file
pdf_file_path = "./Papers/I Am The Passenger - How Visual Motion Cues Can Influence Sickness For In-Car VR.pdf"

Prepromt = """
Provide sentences directly related to hardware to assist the model in understanding the task. Here are some examples:
1. If your input includes any information related to hardware or devices, please list them.
2. Please ensure that the sentences provided are directly related to hardware, but there is no need to express this association in the output.
3. The current task is to enable the model to determine whether the sentence is related to hardware. If it is, please respond with "Yes"; if not, respond with "No".
4. All the sentences provided in this task are excerpts from a research paper.
"""

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
Don't add any additional comments or summaries.
"""

prompt_v3 = """
Is there any sentence related to hardware? If yes, please list all such sentences directly without any additional comments or summaries.
Print out each original sentence directly.
"""

prompt_v4 = """
Please determine whether the following sentences are related to hardware. 
If they are, respond with "Yes"; if not, respond with "No".
"""

prompt_v5 = """
The sentence is from paper.
Please determine whether the following sentences are related to hardware and only hardware.
If they are, respond with "Yes"; if not, respond with "No".
Keep in mind that sentences related to hardware may contain information about devices or technical terms.
"""

prompt_v6 = """
Please determine whether the following sentences are related to hardware and only hardware. If they are, respond with "Yes"; if not, respond with "No".
Keep in mind that sentences related to hardware may contain information about devices or technical terms.
"""

prompt = prompt_v6

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
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error while applying model to sentence: {e}")
        return None

def main():
    start_time = time.time()

    text = extract_text_from_pdf(pdf_file_path)
    if text is None:
        return
    
    sentences = text.split(". ")

    file_name = os.path.basename(pdf_file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    output_file_name = f"{file_name_without_extension}_output.txt"

    print(output_file_name)
    with open(output_file_name, "w", encoding = "utf-8") as f:
        for sentence in sentences:
            response = apply_model_to_sentence(sentence)
            if response is not None and response.strip() == "Yes":
                sentence = sentence.replace("- ", "")
                f.write(sentence.strip() + ".\n")
                print(sentence.strip() + ".")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time: .2f} seconds")

if __name__ == "__main__":
    main()
