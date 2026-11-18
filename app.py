"""Streamlit frontend for the PDF RAG chatbot."""

from __future__ import annotations

import streamlit as st

from rag_backend import answer_question, get_embedding_model, index_documents


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions grounded in their contents.")


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    return get_embedding_model()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = ()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    file_signature = tuple(
        (file.name, file.size) for file in (uploaded_files or [])
    )
    if st.button("Index documents", type="primary", disabled=not uploaded_files):
        try:
            with st.spinner("Extracting text and building the vector index..."):
                count = index_documents(
                    [(file.name, file.getvalue()) for file in uploaded_files],
                    load_embedding_model(),
                )
            st.session_state.indexed_files = file_signature
            st.session_state.messages = []
            st.success(f"Indexed {count} text chunks.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

    if st.session_state.indexed_files:
        st.caption("Indexed files:")
        for name, _ in st.session_state.indexed_files:
            st.write(f"- {name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                answer, chunks = answer_question(question, load_embedding_model())
            sources = sorted({f"{chunk.source}, page {chunk.page}" for chunk in chunks})
            st.markdown(answer)
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- `{source}`")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
