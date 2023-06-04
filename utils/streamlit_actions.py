# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" Streamlit actions for user interaction and session state update."""
# -------------------------------------------
import os
from typing import Any

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.llm import create_memory
from utils.loader import load_csv, load_pdf, load_txt, load_word


def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""


def regenerate_answer():
    st.session_state["regenerate_response"] = True


def initiate_session_state_values():
    values = {
        "new_files": False,
        "docs": [],
        "vectorstore": False,
        "user_question": "",
        "regenerate_response": False,
        "files_metadata": {},
        "chat_history": [],
        "memory": create_memory(),
        "source_docs_history": [],
        "read_files_info": [],
        "vector_search_type": "similarity",
        "vector_k_results": 7,
        "temperature": 0.5,
        "max_tokens_response": 1250,
        "openai_endpoint": "Public OpenAI",
        "OPENAI_API_KEY": "",
    }

    [create_credential(key, value) for key, value in values.items()]


def file_uploader() -> UploadedFile:
    file_processors = {
        ".pdf": load_pdf,
        ".docx": load_word,
        ".csv": load_csv,
        ".txt": load_txt,
    }

    if not st.session_state["OPENAI_API_KEY"] != "":
        st.warning("Please, indicate the OpenAI API KEY in the left menu.")

    files = st.file_uploader(
        "Upload your document(s)",
        accept_multiple_files=True,
        type=list(file_processors.keys()),
    )

    if st.button("Upload file(s)"):
        st.session_state["files_metadata"].clear()
        st.session_state["new_files"] = True
        with st.spinner(f"Uploading file(s) ..."):
            if files is not None:
                for file in files:
                    file_extension = os.path.splitext(file.name)[-1].lower()
                    file_processors[file_extension](file)

    return files


def create_credential(credential_name: str, value: Any):
    if credential_name not in st.session_state:
        st.session_state[credential_name] = value


def save_credential(credential_name: str):
    st.session_state[credential_name] = st.session_state[credential_name]


def ask_credentials(openai_endpoint):
    st.sidebar.text_input(
        type="password",
        placeholder="OPENAI API KEY",
        key="OPENAI_API_KEY",
        label="OPENAI_API_KEY",
        label_visibility="collapsed",
        on_change=save_credential("OPENAI_API_KEY"),
    )

    if not openai_endpoint == "Public OpenAI":
        CREDENTIALS = [
            "OPENAI_API_BASE",
            "OPENAI_API_VERSION",
            "OPENAI_DEPLOYMENT_NAME",
            "OPENAI_ENGINE",
            "OPENAI_EMBEDDINGS_ENGINE",
            "TRANSLATOR_ENDPOINT",
            "TRANSLATOR_API_KEY",
            "TRANSLATOR_SERVICE_LOCATION",
            "LANGUAGE_ENDPOINT",
            "LANGUAGE_API_KEY",
            "LANGUAGE_SERVICE_LOCATION",
        ]
        list(map(lambda x: create_credential(x, ""), CREDENTIALS))
        [
            st.sidebar.text_input(
                placeholder=cred,
                key=cred,
                label=cred,
                label_visibility="collapsed",
                on_change=save_credential(cred),
            )
            for cred in CREDENTIALS
        ]


def sidebar():
    with st.sidebar.container():
        st.title("Hello BASF  :wave:")
        st.subheader("I am Marcos, AI Engineer.")
        st.markdown(
            "[LinkedIn profile](https://www.linkedin.com/in/marcosdatascientist/)"
        )
        st.divider()

        st.sidebar.title("Customise your app")
        settings = st.sidebar.selectbox(
            "Settings", ("Choose OpenAI endpoint", "Configure model features")
        )

        if not settings == "Configure model features":
            st.session_state["openai_endpoint"] = st.sidebar.selectbox(
                "Select OpenAI endpoint", ("Public OpenAI", "Azure OpenAI")
            )

            if st.session_state["openai_endpoint"] != "":
                ask_credentials(st.session_state["openai_endpoint"])

        else:
            st.session_state["vector_search_type"] = st.sidebar.selectbox(
                "Select vector search type", ("similarity", "mmr")
            )
            st.sidebar.markdown("")
            st.session_state["vector_k_results"] = st.sidebar.slider(
                "Select K vector results",
                1,
                15,
                st.session_state["vector_k_results"],
                1,
            )
            st.sidebar.markdown("")
            st.session_state["temperature"] = st.sidebar.slider(
                "Select temperature", 0.0, 1.0, st.session_state["temperature"], 0.1
            )
            st.sidebar.markdown("")
            st.session_state["max_tokens_response"] = st.sidebar.slider(
                "Select max tokens response",
                100,
                2000,
                st.session_state["max_tokens_response"],
                50,
            )
