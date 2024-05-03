# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Our sentences to encode
# sentences = [
#     "This framework generates embeddings for each input sentence",
#     "Sentences are passed as a list of string.",
#     "The quick brown fox jumps over the lazy dog."
# ]

# # Sentences are encoded by calling model.encode()
# embeddings = model.encode(sentences)

# # Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")


import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from ctransformers import AutoModelForCausalLM

model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
# db_Chroma = Chroma.from_documents(docs,model,persist_directory=".\\Embeddings-and-RAG\\db\\chroma_"+"temp")
# db_Chroma.persist()
# db_connection = Chroma(persist_directory=".\\Embeddings-and-RAG\\db\\chroma_"+"temp")

# load it into FAISS
db_FAISS_path = ".\\Embeddings-and-RAG\\db\\faiss_"+"temp"
if os.path.isfile(f"{db_FAISS_path}\\index.faiss"):
    db_FAISS = FAISS.load_local(db_FAISS_path,model,allow_dangerous_deserialization=True)
else:
    # # load the document and split it into chunks
    # loader = TextLoader("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\Embeddings-and-RAG\\FDR_State_of_Union_1944.txt")
    # documents = loader.load()

    # # split it into chunks
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    # docs = text_splitter.split_documents(documents)
    
    docs = []
    loader = TextLoader("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\Embeddings-and-RAG\\Sports Section.txt")
    documents = loader.load()
    docs.append(documents[0])

    loader = TextLoader("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\Embeddings-and-RAG\\Finance Section.txt")
    documents = loader.load()
    docs.append(documents[0])

    # loader = UnstructuredURLLoader(urls="https://blog.tripfez.com/free-things-to-do-in-penang")
    # documents = loader.load()
    # docs.append(documents)

    # loader = UnstructuredURLLoader(urls="https://blog.tripfez.com/12-great-destinations-to-see-in-tokyo")
    # documents = loader.load()
    # docs.append(documents)

    db_FAISS = FAISS.from_documents(docs,model)
    # db_FAISS.save_local(db_FAISS_path)

question = input("What question do you want to ask? ") #"What did FDR say about the cost of food law?"
most_similar_docs = db_FAISS.similarity_search(question)
text = most_similar_docs[0].page_content
title = os.path.basename(most_similar_docs[0].metadata['source'])
# print(most_similar_docs[0].page_content)

prompt = f"""Answer this question:
{question}
Only use below context to answer.
{text}"""

print(prompt)

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