# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" Talk to your data webApp, offering both OpenAI Public API and Azure OpenAI
to interact with own documents. Includes conversation memory."""
# -------------------------------------------
import os
import time

import openai
import streamlit as st
from openai.error import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from utils.llm import (
    build_bot_engine,
    create_vector_db,
    define_embeddings,
    define_llm,
    define_template,
    get_bot_reply,
    get_source_documents,
)
from utils.streamlit_actions import (
    file_uploader,
    initiate_session_state_values,
    regenerate_answer,
    sidebar,
    submit,
)
from utils.tokens import get_total_docs_tokens, tiktoken_len
from utils.translator import translation

st.set_page_config(
    page_title="Talk to your data",
    page_icon="💬",
    layout="wide",
)

st.title("💬  Talk to your data")
st.caption("Interact with your own documents using OpenAI Public API or Azure OpenAI.")
st.markdown("---")

initiate_session_state_values()
file_uploader()
sidebar()

if (
    st.session_state["docs"]
    and st.session_state["new_files"]
    and st.session_state["OPENAI_API_KEY"] != ""
):
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
    openai.api_key = st.session_state["OPENAI_API_KEY"]

    if st.session_state["openai_endpoint"] == "Azure OpenAI":
        openai.api_type = "azure"
        openai.api_version = st.session_state["OPENAI_API_VERSION"]
        openai.api_base = st.session_state["OPENAI_API_BASE"]

    with st.spinner(f"Building embeddings ..."):
        start_embedding_process = time.time()
        embeddings = define_embeddings(
            st.session_state["openai_endpoint"], st.session_state["OPENAI_API_KEY"]
        )
        vectorstore = create_vector_db(st.session_state["docs"], embeddings)
        end_embedding_process = time.time()
        doc_process_time = round(end_embedding_process - start_embedding_process, 2)
        total_tokens = get_total_docs_tokens(st.session_state["docs"])

        st.session_state["vectorstore"] = vectorstore
        st.session_state[
            "read_files_info"
        ] = f"**Time:** {doc_process_time} seconds to create the embeddings\
                                                \n\n**Processed tokens:** {total_tokens}\
                                                \n\n**Total chunks:** {len(st.session_state.docs)}"
        st.session_state["new_files"] = False

if st.session_state["vectorstore"]:
    st.markdown("\n")
    st.write(
        [
            f"{filename}: {values}"
            for filename, values in st.session_state["files_metadata"].items()
        ]
    )
    st.info(st.session_state["read_files_info"])
    st.markdown("---")


with st.form("chat"):
    text_input = st.text_area(
        placeholder="Write your question",
        key="widget",
        label="User question",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 10])
    with col1:
        send_question = st.form_submit_button("Send", on_click=submit)
    with col2:
        regenerate_response = st.form_submit_button(
            "Regenerate response", on_click=regenerate_answer
        )

if (
    (send_question or st.session_state["regenerate_response"])
    and st.session_state.user_question != ""
) and st.session_state["vectorstore"]:
    with st.spinner(f"Processing question ..."):
        llm = define_llm(
            st.session_state["openai_endpoint"],
            st.session_state["OPENAI_API_KEY"],
            st.session_state["max_tokens_response"],
            st.session_state["temperature"],
        )
        prompt = define_template()
        bot_engine = build_bot_engine(
            llm,
            prompt,
            st.session_state["vectorstore"],
            st.session_state["memory"],
            st.session_state["vector_search_type"],
            st.session_state["vector_k_results"],
        )

        try:
            bot_reply = get_bot_reply(
                st.session_state.user_question,
                bot_engine,
            )
            unique_docs = get_source_documents(bot_reply)

        except RateLimitError as err:
            st.error(
                f"Too many requests in a short period of time. Wait a minute an try again.\n\n{err}"
            )

        except InvalidRequestError as err:
            st.error(f"Invalid request, check the model settings.\n\n{err}")

        except APIConnectionError:
            st.error("Error communicating with OpenAI. Connection aborted. Try again.")

        except APIError as err:
            st.error(
                f"Something went wrong on OpenAI side when processing the request. Try again.\n\n{err}"
            )

        except Timeout as err:
            st.error(
                f"The request took too long to complete and OpenAI server closed the connection. Try again.\n\n{err}"
            )

        except AuthenticationError as err:
            st.error(
                f"The API key or token is invalid, expired, or revoked. Check it out.\n\n{err}"
            )

        except ServiceUnavailableError as err:
            st.error(
                f"OpenAI servers are temporarily unable to handle the request. Try again in a few minutes.\n\n{err}"
            )

        except ZeroDivisionError as err:
            st.error(f"Please, write a query before clik the 'Send' button.")

        answer_translation = (
            translation(st.session_state.user_question, bot_reply["answer"])
            if st.session_state["openai_endpoint"] == "Azure OpenAI"
            else bot_reply["answer"]
        )

        st.session_state["chat_history"].append(
            (st.session_state.user_question, answer_translation)
        )

        st.session_state["source_docs_history"].append(unique_docs)

        if st.session_state.regenerate_response:
            st.session_state.regenerate_response = False

if not st.session_state.user_question:
    st.warning("Please, upload your PDF document(s) and/or write your question.")

st.markdown(" ")

for idx, chat in enumerate(st.session_state["chat_history"][::-1]):
    question = chat[0]
    answer = chat[1]
    source_docs = st.session_state["source_docs_history"][::-1][idx]
    question_tokens = tiktoken_len(question)
    answer_tokens = tiktoken_len(answer)

    st.info(question)
    st.caption(f"Tokens: {question_tokens}")

    st.markdown(answer, unsafe_allow_html=True)
    st.caption(f"Tokens: {answer_tokens}")
    st.write(source_docs)
    st.markdown(" ")
    st.markdown(" ")
