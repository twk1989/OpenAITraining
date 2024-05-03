from openai import OpenAI
from ctransformers import AutoModelForCausalLM
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import os
import tiktoken
import pandas as pd
import numpy as np

client = OpenAI()

def get_text_from_pdf_document(file_path):
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Initialize a PDF file reader
            pdf_reader = PdfReader(file)
            # Initialize text variable to store the content of the PDF
            text = ''
            # Iterate through each page in the PDF
            for page_num in range(len(pdf_reader.pages)):
                # Extract text from the page
                text += pdf_reader.pages[page_num].extract_text()
                text = text.replace('\n',' ')

        file_name = os.path.basename(file_path)
        return file_name, text
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_name, ""

def get_text_from_website(url):
    # Define a custom User-Agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36'
    }

    # Send a GET request to the URL with the custom headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find and extract all text from the webpage
        text = soup.get_text()
        text = text.replace('\n',' ')
        return url, text
    else:
        print(f"Failed to retrieve the webpage {url}: {response.status_code}({response.reason})")
        return url, ""

def num_tokens_from_string(string):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text):
    result = client.embeddings.create(
      model='text-embedding-ada-002',
      input=text
    )
    return result.data[0].embedding

def vector_similarity(vec1,vec2):
    return np.dot(np.array(vec1), np.array(vec2))

# Step 1: Get Original Documents
documents = []
sports_news_text = {'title':'Sports Section',
                    'text':"KUALA LUMPUR: Lee Chong Wei says he \"feels like giving up\" on Malaysian badminton and warned ahead of the Olympics that \"drastic\" changes were needed to rescue the sport in the country. \nThe Malaysian badminton great spoke out after his country lost 3-0 to China in the final of the men's competition at the Asia Team Championships in Kuala Lumpur on Sunday.\nChina were far superior despite not fielding their strongest side. India lifted the women's crown, with hosts Malaysia failing to get beyond the quarter-finals.\nThe tournament offered ranking points towards this summer's Paris Olympics."}
documents.append(sports_news_text)

finance_news_text = {'title':"Finance Section",
                     'text':"Intel Corp.’s latest attempt to prove to investors that its turnaround strategy is on track instead spurred a $27 billion rout that further widened the gap between the chipmaker and its peers as it struggles to capitalize on investor demand for all things artificial intelligence-related.\nThe company has plunged 15% since an April 2 update about its newly broken-out foundry division revealed a less-than-rosy outlook for operating losses in the unit. That’s pushed its year-to-date declines to 25%, a stark reversal from its 90% rally in 2023 and well behind strong returns from peers like Nvidia Corp. and Advanced Micro Devices Inc."}
documents.append(finance_news_text)

source, text = get_text_from_website("https://blog.tripfez.com/free-things-to-do-in-penang")
if text != "":
    penang_text = {'title': source,
                    'text': text}
    documents.append(penang_text)

source, text = get_text_from_website("https://blog.tripfez.com/12-great-destinations-to-see-in-tokyo")
if text != "":
    tokyo_text = {'title': source,
                    'text': text}
    documents.append(tokyo_text)

source, text = get_text_from_pdf_document("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\Embeddings-and-RAG\\DDI0429A_ip_exact_components_v1_0_ref_manual.pdf")
if text != "":
    ipxact_text = {'title': source,
                    'text': text}
    documents.append(ipxact_text)

# Step 2: Create and store Vector Embeddings
# result = client.embeddings.create(
#     model='text-embedding-ada-002',
#     input=sports_news_text['text']
# )
# print(result.data[0].embedding)

# encoding = tiktoken.get_encoding("cl100k_base")
# num_tokens = len(encoding.encode(sports_news_text['text']))
# print(num_tokens)

df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
df['Token Count'] = df['Text'].apply(num_tokens_from_string)
df['Embeddings'] = df['Text'].apply(lambda text: get_embedding(text))

# Step 3: Similarity Search
question = input("What question do you want to ask? ")
question_embedding = get_embedding(question)
# result = np.dot(np.array(query_embedding), np.array(df['Embeddings'][0]))
df['prompt_similarity'] = df['Embeddings'].apply(lambda emb: vector_similarity(question_embedding, emb))
df.to_csv('.\\Embeddings-and-RAG\\embeddings_store.csv')

 # get most similar summary
text = df.nlargest(1,'prompt_similarity').iloc[0]['Text']
title = df.nlargest(1,'prompt_similarity').iloc[0]['Title']

# Step 4: Injext Text as Context
prompt = f"""Answer this question:
{question}
Only use below context to answer.
{text}"""

# prompt = f"""Answer this question:
# {question}
# Only use below context to answer. Only answer the question if you have 100% certainty, if no answer, reply 'Invalid'.
# {text}"""

print(prompt)

# response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0

# )
# response_text = response.choices[0].message.content

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model_file = "C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\llama-2-7b.Q4_0.gguf"
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGUF", model_file=model_file, model_type="llama", gpu_layers=0, context_length=8192)
model.config.temperature=0
model.config.context_length = 8192
response_text = model(prompt)

if (response_text == 'Invalid'):
    print(response_text)
else:
    print(f'{response_text}\n\nSource Document:{title}')
