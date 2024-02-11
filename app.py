import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAU6l0P-HY_EjbpDAMuvFxVs9f89JjVWYo"
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS

def load_and_split_pdf(file_path):
    if file_path is not None:
        loader = PyPDFLoader(file_path)
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
    # Save the uploaded file temporarily
    with st.spinner("Processing..."):
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

    # Load and split PDF pages
    pages = load_and_split_pdf(temp_path)

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
        print("Document Content:", doc_content)

        # Perform similarity search for general question
        general_docs = db.similarity_search(general_query)
        general_content = "\n".join([x.page_content for x in general_docs])
        print("General Content:", general_content)

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

        # Uncomment the following lines for actual usage
        result_doc = llm.invoke(doc_input_text)
        result_general = llm.invoke(general_input_text)

        print("Document-based Question Answer:", result_doc.content)
        print("General Question Answer:", result_general.content)

        # Display the results
        st.subheader("Document-based Question Answer:")
        st.write(result_doc.content)

        st.subheader("General Question Answer:")
        st.write(result_general.content)

    # Remove the temporary file
    os.remove(temp_path)
else:
    st.warning("Please upload a PDF file.")
