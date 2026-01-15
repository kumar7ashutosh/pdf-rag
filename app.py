import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question
import streamlit as st
groq_api_key = st.secrets["GROQ_API_KEY"]

working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title("🦙 llama-3.1-8b-instant - Document RAG")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully")

user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):

    answer = answer_question(user_question)

    st.markdown("### llama-3.1-8b-instant Response")
    st.markdown(answer)