import os
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
# import streamlit as st
# groq_api_key = st.secrets["GROQ_API_KEY"]


working_dir = os.path.dirname(os.path.abspath((__file__)))
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

def process_document_to_chroma_db(file_name):
    import shutil

    VECTORSTORE_DIR = "/tmp/doc_vectorstore"

    # 🔥 Always reset vectorstore (Streamlit-safe)
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    loader = PyPDFLoader(os.path.join(working_dir, file_name))
    documents = loader.load()

    if not documents:
        raise RuntimeError("No documents loaded from PDF")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise RuntimeError("Text splitting produced no chunks")

    # ✅ HARD embedding sanity check
    test_embedding = embedding.embed_query("hello world")
    if not test_embedding or len(test_embedding) == 0:
        raise RuntimeError("Embedding model returned empty vector")

    # ✅ Create Chroma
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_DIR
    )

    return True


def answer_question(user_question):
    VECTORSTORE_DIR = "/tmp/doc_vectorstore"

    vectordb = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    response = qa_chain.invoke({"query": user_question})
    return response["result"]
