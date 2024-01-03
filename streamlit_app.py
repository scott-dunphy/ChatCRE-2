#!/usr/bin/env python
# coding: utf-8

from langchain.chat_models import ChatOpenAI
import os
from pathlib import Path
import pinecone
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import glob
import json
from tqdm.autonotebook import tqdm

#loaders = [UnstructuredFileLoader(os.path.join(os.getcwd(),fn)) for fn in list(glob.glob("/Users/scottdunphy/Documents/ODCE/*.pdf"))]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings()

#pinecone.init(
#    api_key=st.secrets["PINECONE_KEY"],  # find at app.pinecone.io
#    environment='us-west4-gcp'  # next to api key in console
#)

pinecone.init(
    api_key='6a666c6c-7238-48bd-b5f6-3b559d156e10',  # find at app.pinecone.io
    environment='asia-southeast1-gcp-free'  # next to api key in console
)

#index_name = 'nareim'
index_name = 'cre'


#docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name, metadata=metadata)
docsearch = Pinecone.from_existing_index(index_name, embeddings)

chain = load_qa_chain(llm, chain_type="stuff")

def text_to_speech(text):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input=text
    )
    pass

response.stream_to_file(speech_file_path)

def chatcre(query):
    docs = docsearch.similarity_search(query, k=5)
    return chain.run(input_documents=docs, question=query)

st.set_page_config(page_title='ChatCRE')
st.title('ChatCRE')

query_input = st.text_input("Ask questions about the leases, property updates, and loan agreement. Be patient while it runs.")
try:
    write_value = chatcre(query_input)
    if st.button("Read Aloud"):
        text_to_speech(text)
    
        # Check if the output.mp3 file exists
        if os.path.isfile('speech.mp3'):
            st.audio('speech.mp3', format='audio/mp3')
        else:
            st.error("Error generating audio.")
except:
    write_value = "Try this prompt: Is there anything concerning in the property updates?"
st.write(write_value)
