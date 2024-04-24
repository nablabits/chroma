"""
Test amazon bedrock embedding function.

These are reverse engineered tests assuming that the embedding function was
working in the first place so we will make sure it keeps working over time.
If you are around and with AWS access you may add a cassette for it using
pytest-vcr.

We based these tests on this documentation:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

from typing import Union
from unittest.mock import MagicMock

import pytest

from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction


@pytest.mark.parametrize(
    "model_name, expected_model_id",
    (
        (None, "amazon.titan-embed-text-v1"),
        ("custom_model", "custom_model"),
    ),
)
def test_amazon_bedrock_embedding_function(
    model_name: Union[str, None], expected_model_id: str
) -> None:
    mock_aws_session = MagicMock()
    mock_client = mock_aws_session.client
    mock_file = MagicMock()
    mock_file.read.return_value = b'{"embedding": [1, 2, 3]}'
    mock_invoke = mock_client.return_value.invoke_model
    mock_invoke.return_value = {"body": mock_file}

    if model_name:
        ef = AmazonBedrockEmbeddingFunction(
            session=mock_aws_session,
            model_name=model_name,
            some_kwarg_we_dont_care="foo",
        )
    else:
        ef = AmazonBedrockEmbeddingFunction(
            session=mock_aws_session,
            some_kwarg_we_dont_care="foo",
        )

    ef(["This is a text"])

    mock_client.assert_called_once_with(
        service_name="bedrock-runtime", some_kwarg_we_dont_care="foo"
    )

    mock_invoke.assert_called_once_with(
        body='{"inputText": "This is a text"}',
        modelId=expected_model_id,
        accept="application/json",
        contentType="application/json",
    )
