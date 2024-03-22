import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.vectorstores import Pinecone as pcvd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from streamlit_chat import message
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings

st.set_page_config(page_title="PDF Pal: Your Document Dialogue Companion :book:", layout="wide")

# Sidebar
with st.sidebar:
    st.title(':book: PDF Pal: Your Document Dialogue Companion ')
    st.markdown(
        """
        ## Welcome to PDF Pal!
        Engage in fascinating conversations with our AI-powered chatbot. Explore the capabilities of LLMs (Large Language Models) as it interacts with your documents in real-time.

        ### Discover More
        - [Streamlit](https://streamlit.io/): Crafted with Streamlit for a seamless user experience.
        - [LangChain](https://python.langchain.com/): Empowered by LangChain for advanced natural language processing.
        - [HuggingFace](https://huggingface.co/): Leveraging open-source cutting-edge LLM model for intelligent responses.
        - [Pinecone](https://www.pinecone.io/): Utilized Pinecone for efficient storage and retrieval of vectors, enhancing data processing capabilities.

        ### Get Started
        - Upload any pdf file under 200MB and let the application process the file.
        - Simply type your message in the chat input box and press Enter to start the conversation.
        - Experience the thrill of conversing with an AI!

        ### Disclaimer
        Please note that this chatbot is for demonstration purposes only and may not always provide accurate or reliable responses.
        """
    )

pc = Pinecone(api_key='YOUR_API_KEY')   
index = pc.Index('chatpdf')

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
            )
    chunks = text_splitter.split_text(text=text)
    return chunks

def embed_and_store(chunks):
    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
    knowledge_base = pcvd.from_texts(chunks, embeddings_model, index_name="chatpdf")
    return knowledge_base

def has_been_processed(file_name):
    """Check if the PDF has already been processed."""
    processed_files = set()
    if os.path.exists("processed_files.txt"):
        with open("processed_files.txt", "r") as file:
            processed_files = set(file.read().splitlines())
    return file_name in processed_files

def mark_as_processed(file_name):
    """Mark the PDF as processed."""
    with open("processed_files.txt", "a") as file:
        file.write(file_name + "\n")

def handle_enter():
    if 'retriever' in st.session_state:
        user_input = st.session_state.user_input
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Please wait..."):
                try:
                    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
                    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.retriever)
                    bot_response = qa.run(user_input)
                    st.session_state.chat_history.append(("Bot", bot_response))
                except Exception as e:
                    st.session_state.chat_history.append(("Bot", f"Error - {e}"))
            st.session_state.user_input = ""


def main():

    load_dotenv()

    st.header("Interact with PDFs: Upload and Chat! 📄💬")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF file", type = "pdf")

    if uploaded_file:
        file_name = uploaded_file.name
        if not has_been_processed(file_name):
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                chunks = get_chunks(text)
                vectordb = embed_and_store(chunks)
                # st.write(chunks)
                st.session_state.retriever = vectordb.as_retriever()
                mark_as_processed(file_name)
                st.success("PDF Processed and Stored!")
                st.session_state.pdf_processed = True
        else:
            if 'retriever' not in st.session_state:
                with st.spinner("Loading existing data..."):
                    index_name = "chatpdf"
                    embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
                    docsearch = pcvd.from_existing_index(index_name, embeddings_model)
                    st.session_state.retriever = docsearch.as_retriever()
                st.info("PDF already processed. Using existing data.")
                st.session_state.pdf_processed = True
    
    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            if speaker == "Bot":
                message(text, key=f"msg-{idx}")
            else:
                message(text, is_user=True, key=f"msg-{idx}")

        st.text_input("Enter your question here:", key="user_input", on_change=handle_enter)

        if st.session_state.user_input:
            handle_enter()

if __name__== '__main__':
    main()