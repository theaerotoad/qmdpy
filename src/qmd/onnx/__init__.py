"""
Optional ONNX-based in-process text embedding and reranking for qmd.
Do not import this package unless the 'onnx' extra dependencies are installed.
"""

from .embedding import EmbeddingGemmaONNX
from .reranker import EttinONNXReranker

__all__ = ["EmbeddingGemmaONNX", "EttinONNXReranker"]