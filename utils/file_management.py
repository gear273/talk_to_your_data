# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" File management and metadata for webApp."""
# -------------------------------------------
import os
from typing import List

from langchain.docstore.document import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.tokens import get_total_docs_tokens


def save_file(filename: str, upload_path: str, file_object: UploadedFile):
    """Saves a file in local disk.

    Args:
        filename (str): file name.
        upload_path (str): local directory to save the file(s).
        file_object (UploadedFile): file(s) uploaded by the user.

    """

    output_file = open(os.path.join(upload_path, filename), "wb")
    output_file.write(file_object)


def get_file_metadata(
    file_object: UploadedFile,
    documents: List[Document],
    splitted_documents: List[Document],
) -> dict:
    """Gets file type, name and number of pages.

    Args:
        file_object (UploadedFile): file(s) uploaded by the user.
        documents (List[Document]): lists of read text.
        splitted_documents (List[Document]): lists of text splitted into chunks.

    Returns:
        dict: file type, name and number of pages.

    """

    size = round((file_object.size / 1024) / 1000, 2)  # MB
    type = file_object.type
    name = file_object.name
    characters = sum(len(item.page_content) for item in documents)
    tokens = get_total_docs_tokens(documents)
    chunks = len(splitted_documents)

    return {
        name: {
            "type": type.split("/")[-1],
            "size (MB)": size,
            "characters": characters,
            "tokens": tokens,
            "chunks": chunks,
        }
    }


def update_files_metadata(
    files_metadata: dict,
    file_object: UploadedFile,
    documents: List[Document],
    splitted_documents: List[Document],
):
    """Updates file(s) metadata.

    Args:
        files_metadata (dict): file type, name and number of pages.
        file_object (UploadedFile): file(s) uploaded by the user.
        documents (List[Document]): lists of read text.
        splitted_documents (List[Document]): lists of text splitted into chunks.

    """

    file_metadata = get_file_metadata(file_object, documents, splitted_documents)
    files_metadata.update(file_metadata)
