import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests import HTTPError
from requests.exceptions import ConnectionError

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


def test_ollama_with_model() -> None:
    """
    Test an actual embedding provided a server is up and running.

    To set up the Ollama server, follow instructions at:
    https://github.com/ollama/ollama?tab=readme-ov-file

    Then, export the OLLAMA_SERVER_URL and OLLAMA_MODEL environment variables.
    """
    server_url = os.environ.get("OLLAMA_SERVER_URL")
    model_name = os.environ.get("OLLAMA_MODEL")

    if server_url is None or model_name is None:
        pytest.skip(
            "OLLAMA_SERVER_URL or OLLAMA_MODEL environment variable not set. "
            "Skipping test."
        )

    try:
        response = requests.get(server_url)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except (HTTPError, ConnectionError):
        pytest.skip("Ollama server not running. Skipping test.")

    ef = OllamaEmbeddingFunction(model_name=model_name, url=f"{server_url}/embeddings")
    embeddings = ef(["Here is an article about llamas...", "this is another article"])
    assert len(embeddings) == 2


def test_ollama_no_model_is_called_once() -> None:
    ef = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://debian.org/",
    )

    response = requests.Response()
    response._content = b'{"embedding": [1, 2, 3]}'
    response.status_code = 200
    rv = MagicMock(return_value=response)()

    with patch("requests.Session.post", return_value=rv) as mock_request:
        embeddings = ef(["This is a test string"])

    mock_request.assert_called_once_with(
        "http://debian.org/",
        json={"model": "nomic-embed-text", "prompt": "This is a test string"},
    )

    assert embeddings == [[1, 2, 3]]


def test_ollama_no_model_is_called_several_times() -> None:
    ef = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://debian.org/",
    )

    response = requests.Response()
    response._content = b'{"embedding": [1, 2, 3]}'
    response.status_code = 200
    rv = MagicMock(return_value=response)()

    with patch("requests.Session.post", return_value=rv) as mock_request:
        embeddings = ef(["This is a test string", "and this is another"])

    assert mock_request.call_count == 2
    assert embeddings == [[1, 2, 3], [1, 2, 3]]
