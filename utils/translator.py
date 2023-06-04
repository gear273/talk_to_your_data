# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
"""Detects language and translates text using Azure Cognitive services."""
# -------------------------------------------
import logging

import streamlit as st
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

LOGGER = logging.getLogger()


def language_detector(text: str) -> str:
    """Detects the language from a given text.

    Args:
        text (str): text to detect its language.

    Returns:
        str: iso6391 language code.

    """

    text_analytics_client = TextAnalyticsClient(
        endpoint=st.session_state["LANGUAGE_ENDPOINT"],
        credential=AzureKeyCredential(st.session_state["LANGUAGE_API_KEY"]),
    )
    result = text_analytics_client.detect_language(text)
    return result[0].primary_language.iso6391_name


def translator(answer: str, target_language: str) -> str:
    """Translates the bot answer to the question language.

    Args:
        answer (str): bot answer.
        target_language (str): output language.

    Returns:
        str: translated answer.

    """

    credential = TranslatorCredential(
        st.session_state["TRANSLATOR_API_KEY"],
        st.session_state["TRANSLATOR_SERVICE_LOCATION"],
    )
    text_translator = TextTranslationClient(
        endpoint=st.session_state["TRANSLATOR_ENDPOINT"], credential=credential
    )

    answer_translation = text_translator.translate(
        content=[{"text": answer}], to=[target_language]
    )
    answer_translation = (
        answer_translation[0].translations[0].text if answer_translation else answer
    )

    return answer_translation


def translation(question: str, answer: str) -> str:
    """Translates the bot answer to the question language.

    Args:
        question (str): user question.
        answer (str): bot answer.

    Returns:
        str: translated answer.

    """

    try:
        question_language = language_detector([question])
        answer_language = language_detector([answer])
        LOGGER.info(
            f"## LANGUAGES DETECTED: '{question_language}' - '{answer_language}'"
        )

        if not question_language == answer_language:
            answer_translation = translator(answer, question_language)
            LOGGER.info(
                f"## TRANSLATION COMPLETED: '{answer_language}' to '{question_language}' ##"
            )
            return answer_translation

    except HttpResponseError as exception:
        st.error(f"Error Code: {exception.error.code}")
        st.error(f"Message: {exception.error.message}")

    return answer
