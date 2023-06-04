# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" LLM processes for GPT model creation and interaction."""
# -------------------------------------------
from collections import defaultdict

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS


def define_embeddings(
    openai_endpoint: str, openai_api_key: str, chunk_size: int = 1
) -> OpenAIEmbeddings:
    """Defines the embeddings process for text vectorisation.

    Args:
        openai_endpoint (str): Public OpenAI or Azure OpenAI
        openai_api_key (str): OpenAI API KEY.
        chunk_size (int, optional): how many chunks at a time. Defaults to 1.

    Returns:
        OpenAIEmbeddings: embeddings process.

    """

    if openai_endpoint == "Azure OpenAI":
        embeddings = OpenAIEmbeddings(
            model=st.session_state["OPENAI_EMBEDDINGS_ENGINE"],
            chunk_size=chunk_size,
            max_retries=3,
        )

    else:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    return embeddings


def define_llm(
    openai_endpoint: str,
    openai_api_key: str,
    max_tokens: int = 1250,
    temperature: float = 0.5,
) -> AzureChatOpenAI:
    """Defines the llm for chat.

    Args:
        openai_endpoint (str): Public OpenAI or Azure OpenAI
        openai_api_key (str): OpenAI API KEY.
        max_tokens (int, optional): max tokens for chat consumption.
        temperature (float, optional): creativity range for model. 0 for deterministic, 1 for creative.

    Returns:
        AzureChatOpenAI: large language model template.

    """

    if openai_endpoint == "Azure OpenAI":
        llm = AzureChatOpenAI(
            deployment_name=st.session_state["OPENAI_DEPLOYMENT_NAME"],
            model_name=st.session_state["OPENAI_ENGINE"],
            openai_api_version=st.session_state["OPENAI_API_VERSION"],
            openai_api_base=st.session_state["OPENAI_API_BASE"],
            openai_api_key=st.session_state["OPENAI_API_KEY"],
            openai_api_type="azure",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    else:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return llm


def define_template() -> PromptTemplate:
    """Defines the role for the GPT bot, as well as the model prompt template.

    Returns:
        PromptTemplate: role and prompt template for model interaction.

    """

    prompt_template = """You're TalkToYourData-Bot, a world-class expert in the given CONTEXT below.
    Your answers are as much descriptive as possible because you help users to understand the content.
    If you're asked something which is not indicated or related in the CONTEXT, say you don't know.
    Do not try to make up an answer even if you have the information in your general knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER in markdown format:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return prompt


def create_vector_db(docs: list, embeddings: OpenAIEmbeddings) -> FAISS:
    """Converts text into vectors.

    Args:
        docs (list): texts extracted from .pdf files.
        embeddings (OpenAIEmbeddings): embeddings process.

    Returns:
        FAISS: vectorised texts.

    """

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    return vectorstore


def create_memory() -> ConversationBufferMemory:
    """Creates memory object to manage conversation history.

    Returns:
        ConversationBufferMemory: buffer memory.

    """

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return memory


def build_bot_engine(
    llm: AzureChatOpenAI,
    prompt: PromptTemplate,
    vectorstore: FAISS,
    memory: ConversationBufferMemory,
    search_type: str = "similarity",
    k_results: int = 7,
) -> ConversationalRetrievalChain:
    """Builds the GPT bot.

    Args:
        llm (AzureChatOpenAI): large language model template.
        prompt (PromptTemplate): role and prompt template for model interaction.
        vectorstore (FAISS): vectorised texts.
        memory (ConversationBufferMemory): buffer memory.
        search_type (str, optional): embedding vectorstore search type.
        k_results (int, optional): number of embedding vectorstore retrieved results.

    Returns:
        ConversationalRetrievalChain: GPT bot engine ready for interaction.

    """

    bot_engine = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k_results}
        ),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        memory=memory,
    )

    return bot_engine


def get_bot_reply(user_input: str, bot_engine: ConversationalRetrievalChain) -> dict:
    """Gets the bot reply given an user question.

    Args:
        user_input (str): user question.
        bot_engine (ConversationalRetrievalChain): GPT bot engine ready for interaction.

    Returns:
        dict: bot answer.

    """

    return bot_engine({"question": user_input})


def get_source_documents(bot_reply: dict) -> dict:
    """Gets source information the bot used to provide an answer.
    These sources were obtained by cosine similarity comparing user question embedding
    with database embeddings.

    Args:
        bot_reply (dict): bot answer.

    Returns:
        dict: document(s) name(s) and its source page(s).

    """

    unique_docs = defaultdict(list)
    unique_sources = set(
        (
            source.metadata["source"].replace("/", "\\").split("\\")[-1],
            source.metadata.get(
                "page",
                source.metadata.get("row", "info found in file."),
            ),
        )
        for source in bot_reply["source_documents"]
    )

    for doc_filename, doc_info in unique_sources:
        unique_docs[doc_filename].append(
            f"page {doc_info}" if isinstance(doc_info, int) else doc_info
        )

    return dict(unique_docs)
