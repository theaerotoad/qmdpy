"""
Phase 1: Document Mapping & Analysis

This module is responsible for parsing a Markdown document into a list of
"semantic blocks," where each block consists of a header and its content.
"""
import re
from typing import List, Tuple, Optional
from .models import SemanticBlock

# Regex to match images: ![alt text](url) -> capture 'alt text'
# We handle images before links to avoid confusion since images start with !
IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')

# Regex to match links: [text](url) -> capture 'text'
# Negative lookbehind (?<!\!) ensures we don't match images that somehow slipped through
LINK_PATTERN = re.compile(r'(?<!\!)\[([^\]]+)\]\([^\)]+\)')

def parse_markdown_to_blocks(
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    strip_links: bool = True
) -> Tuple[List[SemanticBlock], bool]:
    """
    Parses markdown text or file into a list of SemanticBlock objects.

    Args:
        file_path: The path to the markdown file (if content is not directly supplied).
        content: Direct markdown content string.
        strip_links: If True, converts [text](url) to just 'text'.

    Returns:
        A tuple containing a list of SemanticBlock objects and a boolean
        indicating if any tables were found in the document.
    """
    blocks: List[SemanticBlock] = []
    has_tables = False

    if content is not None:
        lines = content.splitlines(keepends=True)
    elif file_path is not None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (FileNotFoundError, UnicodeDecodeError):
            print(f"Error: Could not read file at {file_path}")
            return [], False
    else:
        return [], False

    current_block_content = []
    current_header_level = 0
    current_header_text = "Preface"
    current_start_line = 1
    in_code_block = False

    header_pattern = re.compile(r'^(#+)\s+(.*)')
    table_pattern = re.compile(r'^\s*\|.*\|')

    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        
        # Apply link stripping if requested, but NOT inside code blocks
        processed_line = line
        if strip_links and not in_code_block:
            # 1. Replace images with their alt text (or empty string if no alt)
            processed_line = IMG_PATTERN.sub(r'\1', processed_line)
            # 2. Replace links with their text
            processed_line = LINK_PATTERN.sub(r'\1', processed_line)

        header_match = header_pattern.match(processed_line)
        if header_match and not in_code_block:
            if current_block_content:
                blocks.append(SemanticBlock(
                    header_level=current_header_level,
                    header_text=current_header_text,
                    content="".join(current_block_content).strip(),
                    start_line=current_start_line
                ))
            
            current_header_level = len(header_match.group(1))
            current_header_text = header_match.group(2).strip()
            current_block_content = [processed_line]
            current_start_line = line_num
        else:
            current_block_content.append(processed_line)
            if table_pattern.match(processed_line):
                has_tables = True

    if current_block_content:
        blocks.append(SemanticBlock(
            header_level=current_header_level,
            header_text=current_header_text,
            content="".join(current_block_content).strip(),
            start_line=current_start_line
        ))

    return blocks, has_tables
