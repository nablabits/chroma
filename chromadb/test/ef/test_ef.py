from chromadb.utils import embedding_functions


def test_get_builtins_holds() -> None:
    """
    Ensure that `get_builtins` is consistent after the ef migration.

    This test is intended to be temporary until the ef migration is complete as
    these expected builtins are likely to grow as long as users add new
    embedding functions.

    The hardcoded list of builtins was generated by running `get_builtins()`
    on this commit: df65e5a65628ef9231f67ccc748a7d6b114c9c02
    """
    expected_builtins = {
        "AmazonBedrockEmbeddingFunction",
        "CohereEmbeddingFunction",
        "GoogleGenerativeAiEmbeddingFunction",
        "GooglePalmEmbeddingFunction",
        "GoogleVertexEmbeddingFunction",
        "HuggingFaceEmbeddingFunction",
        "HuggingFaceEmbeddingServer",
        "InstructorEmbeddingFunction",
        "JinaEmbeddingFunction",
        "ONNXMiniLM_L6_V2",
        "OllamaEmbeddingFunction",
        "OpenAIEmbeddingFunction",
        "OpenCLIPEmbeddingFunction",
        "RoboflowEmbeddingFunction",
        "SentenceTransformerEmbeddingFunction",
        "Text2VecEmbeddingFunction",
        "ChromaLangchainEmbeddingFunction",
    }

    assert expected_builtins == embedding_functions.get_builtins()
