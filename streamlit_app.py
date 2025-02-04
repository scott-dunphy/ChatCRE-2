#!/usr/bin/env python
# coding: utf-8

import os
from io import BytesIO
from pathlib import Path

import pinecone
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# Remove deprecated index creators and outdated LLM imports
# Instead, import the latest Pinecone vector store from its dedicated package
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

import glob
# We'll alias the openai package as tts_client for our text-to-speech functionality.
import openai as tts_client

# (Optional) If you wish to load documents from files:
# loaders = [UnstructuredFileLoader(os.path.join(os.getcwd(), fn)) for fn in glob.glob("/path/to/*.pdf")]

# Set your API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the Chat model and embedding function
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# Initialize Pinecone – note that your API key and environment must be correctly set.
pinecone.init(
    api_key='6a666c6c-7238-48bd-b5f6-3b559d156e10',  # find at app.pinecone.io
    environment='asia-southeast1-gcp-free'            # next to API key in your console
)
index_name = 'cre'

# Create a Pinecone vector store from an existing index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)   [oai_citation_attribution:0‡api.python.langchain.com](https://api.python.langchain.com/en/latest/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html)

# Build a RetrievalQA chain directly from the chain type,
# so we don’t have to manually call similarity_search followed by chain.run.
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

def text_to_speech(text: str):
    """Convert text to speech using OpenAI’s audio API."""
    speech_file_path = "speech.mp3"
    response = tts_client.Audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)

def chatcre(query: str) -> str:
    """Run the QA chain to answer a query over our documents."""
    return qa.run(query)

# Set up the Streamlit app
st.set_page_config(page_title='ChatCRE')
st.title('ChatCRE')

query_input = st.text_input("Ask questions about the leases, property updates, and loan agreement. Be patient while it runs.")

if query_input:
    try:
        result_text = chatcre(query_input)
        st.write(result_text)
        if st.button("Read Aloud"):
            text_to_speech(result_text)
            if os.path.isfile('speech.mp3'):
                st.write("OK to Play!")
                with open('speech.mp3', 'rb') as f:
                    audio_bytes = f.read()
                st.download_button(
                    label='Download Audio',
                    data=BytesIO(audio_bytes),
                    file_name='speech.mp3',
                    mime='audio/mp3'
                )
            else:
                st.error("Error generating audio.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Try this prompt: Is there anything concerning in the property updates?")
