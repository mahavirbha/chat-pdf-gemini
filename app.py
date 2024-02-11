
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS

def load_and_split_pdf(file):
    if file is not None:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        return pages
    else:
        return []

# Streamlit app layout
st.title("Document Q&A App")

# File uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Load and split PDF pages
    pages = load_and_split_pdf(uploaded_file)

    # Initialize embeddings and create FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(pages, embeddings)

    # User input for the document-based question
    doc_query = st.text_input("Enter a question about the document:")

    # User input for the general question
    general_query = st.text_input("Enter a general question:")

    # Search button
    if st.button("Search"):
        # Perform similarity search for document-based question
        doc_docs = db.similarity_search(doc_query)
        doc_content = "\n".join([x.page_content for x in doc_docs])

        # Perform similarity search for general question
        general_docs = db.similarity_search(general_query)
        general_content = "\n".join([x.page_content for x in general_docs])

        # Display context and user questions
        st.subheader("Document-based Question:")
        st.write("Context:\n" + doc_content)
        st.write("User Question:\n" + doc_query)

        st.subheader("General Question:")
        st.write("Context:\n" + general_content)
        st.write("User Question:\n" + general_query)

        # Use Google Generative AI to answer the questions
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        doc_input_text = "Document-based Question Context:" + doc_content + "\nUser question:" + doc_query
        general_input_text = "General Question Context:" + general_content + "\nUser question:" + general_query

        doc_result = llm.invoke(doc_input_text)
        general_result = llm.invoke(general_input_text)

        # Display the results
        st.subheader("Document-based Question Answer:")
        st.write(doc_result.content)

        st.subheader("General Question Answer:")
        st.write(general_result.content)
else:
    st.warning("Please upload a PDF file.")
