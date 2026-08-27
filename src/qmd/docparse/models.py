"""
Data classes for Blocks, Chunks, and the final JSON structure.
Using standard library dataclasses for clarity and type safety.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SemanticBlock:
    """Intermediate representation of a header and its content."""
    header_level: int
    header_text: str
    content: str
    start_line: int
    joiner_to_next: str = "\n\n"

    @property
    def char_count(self) -> int:
        return len(self.content)

@dataclass
class Chunk:
    """Final representation of a single chunk of content."""
    chunk_id: int
    content: str
    start_line_number: int
    character_count: int
    parent_headers: Dict[str, str]
    joiner_to_next: str = "\n\n"
    preceding_table_header: Optional[str] = None
    preceding_codeblock_header: Optional[str] = None
    embedding: Optional[str] = None

@dataclass
class InferredAttributes:
    title: Optional[str] = None
    author: Optional[str] = None
    summary: Optional[str] = None
    doc_date: Optional[str] = None

@dataclass
class DocumentStructure:
    level: int
    text: str
    start_line: int
    chunk_id: int

@dataclass
class ChunkingParameters:
    max_chunk_size: int
    target_chunk_size: int

@dataclass
class EmbeddingSettings:
    model_name: str = "text-embedding-ada-002"

@dataclass
class Metadata:
    source_filename: str
    date_indexed: int
    file_signature: Optional[str] = None
    has_tables: bool = False
    doc_date: Optional[str] = None
    inferred_attributes: InferredAttributes = field(default_factory=InferredAttributes)
    document_structure: List[DocumentStructure] = field(default_factory=list)
    chunking_parameters: Optional[ChunkingParameters] = None
    embedding_settings: EmbeddingSettings = field(default_factory=EmbeddingSettings)

@dataclass
class ChunkerOutput:
    """The final JSON output structure."""
    metadata: Metadata
    chunks: List[Chunk]

