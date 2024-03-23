# PDF Pal: Your Document Dialogue Companion

PDF Pal is a Python application that serves as your trusted companion in managing and interacting with PDF documents. By seamlessly integrating LangChain and Pinecone technologies, PDF Pal offers a sophisticated yet user-friendly platform for navigating through the content of your PDF files.

With PDF Pal, users can engage in natural language conversations, effortlessly posing questions and queries about their PDF files. Leveraging the powerful language model capabilities of LangChain, the application provides accurate and contextually relevant responses, empowering users to extract insights and glean information from their documents with ease.

## Key Features:

- **Intuitive Interface**: PDF Pal boasts an intuitive interface designed to streamline the document exploration process. Navigate through your PDF files effortlessly and discover valuable insights with just a few clicks.

- **Advanced Search Functionality**: Say goodbye to endless scrolling and manual searches. PDF Pal harnesses the power of Pinecone vector database to enable advanced similarity searches, allowing users to quickly locate relevant information based on their queries.

- **Interactive Conversations**: Interact with your PDF documents in a whole new way. Engage in natural language conversations with PDF Pal, asking questions and receiving insightful responses in real-time.

- **Effortless Installation**: Getting started with PDF Pal is a breeze. Simply install the required dependencies using pip and launch the application with Streamlit – it's that easy!

## How it works

The application follows these steps to provide responses to your questions:

1. **PDF Loading and Text Extraction**: The app reads PDF documents and extracts their text content using PyPDF2.

2. **Text Chunking**: The extracted text is divided into smaller chunks that can be processed effectively.

3. **Text Embeddings**: The chunks are converted into embeddings (vectors) using InstructorEmbeddings and are stored in a vector database (Pinecone) for efficient retrieval.

4. **Similarity Matching**: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. **Response Generation**: The selected chunks are passed to the language model (Google Flan T5 xxl), which generates a response based on the relevant content of the PDFs.

## Installation and Usage

To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running the following command:
    ```pip install -r requirements.txt```
3. Obtain an API key from Huggingface Hub and Pinecone and add it to the .env file in the project directory.
4. Run the app.py file using ```streamlit run app.py```

**Usage**

To use the App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the .env file.
2. Run the main.py file using the Streamlit CLI. Execute the following command:
    ```bash
    streamlit run app.py
    ```
3. The application will launch in your default web browser, displaying the user interface.
4. Load multiple PDF documents into the app by following the provided instructions.
5. Ask questions in natural language about the loaded PDFs using the chat interface.

Experience the future of document management with PDF Pal – your ultimate document dialogue companion. Unlock the potential of your PDF files and embark on a journey of discovery today!