# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" Tokens counter module."""
# -------------------------------------------
from typing import List

import streamlit as st
import tiktoken


def tiktoken_len(text: str) -> int:
    """Gets the token consmption of a given text based on the model in use.

    Args:
        text (str): text.

    Returns:
        int: text token consumption.

    """

    openai_engine = (
        "gpt-3.5-turbo"
        if st.session_state["openai_endpoint"] == "Public OpenAI"
        else st.session_state["OPENAI_ENGINE"]
    )
    encoding = tiktoken.encoding_for_model(openai_engine)
    tokenizer = tiktoken.get_encoding(encoding.name)
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_total_docs_tokens(docs: List[list]) -> int:
    """Gets potential total token consumption of the provided documents.

    Args:
        docs (List[list]): text read per .pdf file in a given directory.

    Returns:
        int: total token consumption.

    """

    total_tokens = 0

    for doc in docs:
        doc_tokens = tiktoken_len(doc.page_content)
        total_tokens += doc_tokens

    return total_tokens
