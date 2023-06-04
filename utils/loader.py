# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" File(s) loaders to read and collect text."""
# -------------------------------------------
import os
import tempfile
from typing import Union

import streamlit as st
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.file_management import save_file, update_files_metadata
from utils.tokens import tiktoken_len


def create_text_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    """Creates text splitter to divide the read text.

    Args:
        chunk_size (int): max tokens per chunk.
        chunk_overlap (int): the max overlap between chunks.
                             Recommended some overlap to maintain some continuity between chunks.

    Returns:
        RecursiveCharacterTextSplitter: text splitter ready for use.

    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter


def file_loader(
    file: str,
    loader_class: Union[CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader],
    chunk_size: int = 280,
    chunk_overlap: int = 40,
):
    """Loads .pdf files in a given directory and extracts its content.

    Args:
        dir_path (str): directory path.
        loader_class (multiple): loader for specific file type.
        chunk_size (int, optional): max tokens per chunk.
        chunk_overlap (int, optional): the max overlap between chunks.
                             Recommended some overlap to maintain some continuity between chunks.

    Returns:
        list: text read per file in the given directory.

    """

    text_splitter = create_text_splitter(chunk_size, chunk_overlap)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_file(file.name, temp_dir, file.getvalue())

        loader = loader_class(os.path.join(temp_dir, file.name))
        documents = loader.load()
        splitted_documents = text_splitter.split_documents(documents)
        st.session_state["docs"] += splitted_documents
        update_files_metadata(
            st.session_state["files_metadata"], file, documents, splitted_documents
        )


def load_pdf(file: UploadedFile) -> file_loader:
    return file_loader(file, PyPDFLoader)


def load_word(file: UploadedFile) -> file_loader:
    return file_loader(file, Docx2txtLoader)


def load_csv(file: UploadedFile) -> file_loader:
    return file_loader(file, CSVLoader)


def load_txt(file: UploadedFile) -> file_loader:
    return file_loader(file, TextLoader)
