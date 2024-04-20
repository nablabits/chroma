from unittest.mock import MagicMock, patch

import pytest

from chromadb.utils.embedding_functions import CohereEmbeddingFunction


def test_cohere_does_not_find_library() -> None:
    with pytest.raises(ValueError) as e:
        CohereEmbeddingFunction(api_key="test", model_name="large")

    assert e.value.args[0] == (
        "The cohere python package is not installed. Please install it with "
        "`pip install cohere`"
    )


def test_cohere_calls_embed() -> None:
    cohere = MagicMock()
    cohere.Client = MagicMock()
    cohere.Client.return_value.embed.return_value = [[1, 2, 3]]

    with patch("importlib.import_module", return_value=cohere) as cohere_mock:
        ef = CohereEmbeddingFunction(api_key="test", model_name="large")
        embeddings = ef(["This is a text"])

    cohere_mock.return_value.Client.assert_called_once_with("test")
    cohere_mock.return_value.Client.return_value.embed.assert_called_once_with(
        texts=["This is a text"],
        model="large",
        input_type="search_document",
    )

    assert ef._model_name == "large"
    assert embeddings == [[1, 2, 3]]
