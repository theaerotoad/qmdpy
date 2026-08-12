"""
Phase 2: Optimized Grouping & Chunking

This module takes the map of semantic blocks from the parser and groups them
into the final, well-balanced chunks using an optimization algorithm to
minimize the variance in chunk sizes.
"""

import re
from typing import List, Dict, Optional
from .models import SemanticBlock, Chunk

# Regex to identify a markdown table row or code block fence
table_pattern = re.compile(r'^\s*\|.*\|')
codeblock_pattern = re.compile(r'^\s*```(\w*)')

def _extract_table_header(content: str) -> Optional[str]:
    """
    Extracts the first two lines of the first Markdown table (header and separator)
    found anywhere in the content.
    """
    lines = content.split('\n')
    # A valid header requires at least two lines, so iterate up to the second-to-last line
    for i in range(len(lines) - 1):
        header_line = lines[i]
        separator_line = lines[i+1]

        # Check if both lines look like table rows and the separator is valid
        if table_pattern.match(header_line) and \
           table_pattern.match(separator_line) and \
           '---' in separator_line:
            # We found the first valid table header in this chunk
            return f"{header_line}\n{separator_line}"
    
    # No table header found in the content
    return None

def _extract_codeblock_header(content: str) -> Optional[str]:
    """
    Extracts the opening fence and language of the first code block found.
    """
    for line in content.split('\n'):
        if codeblock_pattern.match(line):
            return line
    return None

def _split_oversized_block(block: SemanticBlock, max_size: int) -> List[SemanticBlock]:
    """
    Splits a single semantic block if its content exceeds the max size.
    This function identifies the precise separator used for splitting (`\n\n`, `\n`, `. `, etc.)
    and stores it so the document can be perfectly reconstructed.
    """
    if block.char_count <= max_size:
        return [block]

    sub_blocks = []
    header_pattern = re.compile(r'^(#+\s+.*\n*)')
    header_match = header_pattern.match(block.content)
    header = header_match.group(0) if header_match else ""
    content_body = block.content[len(header):].lstrip()
    
    is_first_sub_block = True
    offset = 0
    while offset < len(content_body):
        effective_max_size = max_size - (len(header) if is_first_sub_block else 0)
        split_point = -1
        joiner = ""

        if len(content_body) - offset > effective_max_size:
            # Find the best split point from right to left within the allowed chunk size.
            # The hierarchy of preference is:
            # 1. Paragraph break ('\n\n')
            # 2. Line break ('\n')
            # 3. Sentence break ('. ', '? ', '! ')
            # 4. Word break (' ')
            # 5. Hard cut (last resort)

            # 1. Paragraph break
            p_split = content_body.rfind('\n\n', offset, offset + effective_max_size)
            if p_split > offset:
                split_point = p_split
                joiner = '\n\n'
            
            # 2. Line break
            if split_point == -1:
                l_split = content_body.rfind('\n', offset, offset + effective_max_size)
                if l_split > offset:
                    split_point = l_split
                    joiner = '\n'

            # 3. Sentence break (if no paragraph or line break)
            if split_point == -1:
                sentence_enders = ['. ', '? ', '! ']
                best_s_split = -1
                for ender in sentence_enders:
                    s_split = content_body.rfind(ender, offset, offset + effective_max_size)
                    # We want the right-most sentence break we can find
                    if s_split > best_s_split:
                        best_s_split = s_split
                
                if best_s_split > offset:
                    split_point = best_s_split + 1  # Split point is after the punctuation
                    joiner = ' '                    # Joiner is the space after the punctuation
            
            # 4. Word break (if nothing else)
            if split_point == -1:
                w_split = content_body.rfind(' ', offset, offset + effective_max_size)
                if w_split > offset:
                    split_point = w_split
                    joiner = ' '
            
            # 5. Hard cut (last resort)
            if split_point == -1:
                split_point = offset + effective_max_size
                joiner = ''
        
        end_point = split_point if split_point != -1 else len(content_body)
        chunk_text = content_body[offset:end_point]
        
        final_content = (header + chunk_text).strip() if is_first_sub_block else chunk_text.strip()
        header_text_to_use = block.header_text if is_first_sub_block else f"{block.header_text} (cont.)"
        
        if final_content:
            sub_blocks.append(SemanticBlock(
                header_level=block.header_level,
                header_text=header_text_to_use,
                content=final_content,
                start_line=block.start_line,
                joiner_to_next=joiner
            ))
        
        is_first_sub_block = False
        offset = end_point + len(joiner)

    # The last sub-block should inherit the joiner of the original block.
    if sub_blocks:
        sub_blocks[-1].joiner_to_next = block.joiner_to_next

    return sub_blocks


def _calculate_parent_headers(original_blocks: List[SemanticBlock], current_block: SemanticBlock) -> Dict[str, str]:
    """Calculates the hierarchical parent headers for a given block."""
    parents = {}
    current_index = -1
    for i, b in enumerate(original_blocks):
        if b.start_line == current_block.start_line and b.header_text in current_block.header_text:
            current_index = i
            break
            
    if current_index == -1: return {}

    current_original_block = original_blocks[current_index]
    current_level = current_original_block.header_level
    if current_level > 0:
         parents[str(current_level)] = f"{'#' * current_level} {current_original_block.header_text}"

    for i in range(current_index - 1, -1, -1):
        prev_block = original_blocks[i]
        if prev_block.header_level < current_level:
            parents[str(prev_block.header_level)] = f"{'#' * prev_block.header_level} {prev_block.header_text}"
            current_level = prev_block.header_level
            if current_level == 1: break
                
    return dict(sorted(parents.items()))

def _create_final_chunks(partitions: List[List[SemanticBlock]], original_blocks: List[SemanticBlock]) -> List[Chunk]:
    """Assembles the final list of Chunk objects from the partitions."""
    potential_chunks: List[Chunk] = []
    chunk_id_counter = 0

    # First pass: create all chunks without cross-chunk context
    for partition in partitions:
        if not partition: continue
        
        content_parts = []
        for i, b in enumerate(partition):
            content_parts.append(b.content)
            if i < len(partition) - 1:
                content_parts.append(b.joiner_to_next)
        full_content = "".join(content_parts)

        first_block = partition[0]
        last_block = partition[-1]
        
        parent_headers = _calculate_parent_headers(original_blocks, first_block)
        
        headers_to_remove = [lvl for lvl, txt in parent_headers.items() if full_content.strip().startswith(txt.strip())]
        for level in headers_to_remove:
            del parent_headers[level]

        chunk = Chunk(
            chunk_id=chunk_id_counter,
            content=full_content,
            start_line_number=first_block.start_line,
            character_count=len(full_content),
            parent_headers=parent_headers,
            joiner_to_next=last_block.joiner_to_next,
        )
        potential_chunks.append(chunk)
        chunk_id_counter += 1

    # Second pass: iterate through chunks to add table and codeblock context
    in_code_block = False
    current_code_block_header = None
    for i, current_chunk in enumerate(potential_chunks):
        # --- Carry-forward context from previous chunk ---
        if i > 0:
            previous_chunk = potential_chunks[i-1]
            # Table context: Check if previous chunk ended in a table and current one starts with one
            if table_pattern.match(current_chunk.content.lstrip()):
                last_line_of_prev = previous_chunk.content.rstrip().split('\n')[-1]
                if table_pattern.match(last_line_of_prev):
                    header = None
                    for j in range(i - 1, -1, -1):
                        header = _extract_table_header(potential_chunks[j].content)
                        if header:
                            break
                    if header:
                        current_chunk.preceding_table_header = header
            # Code block context: Apply header if we are in a continued block
            if in_code_block:
                current_chunk.preceding_codeblock_header = current_code_block_header

        # --- Update state based on the current chunk's content ---
        fence_count = current_chunk.content.count('```')
        if fence_count > 0:
            # If we are not in a code block, the first fence we see is a potential start
            if not in_code_block:
                current_code_block_header = _extract_codeblock_header(current_chunk.content)
            
            # An odd number of fences flips the state
            if fence_count % 2 != 0:
                in_code_block = not in_code_block
            
            # If we just exited a code block, clear the header so it's not used for the next chunk
            if not in_code_block:
                current_code_block_header = None


    return potential_chunks

def _greedy_fallback(blocks: List[SemanticBlock], target_chunk_size: int, max_chunk_size: int) -> List[List[SemanticBlock]]:
    """A simple greedy partitioning method used as a last resort."""
    partitions: List[List[SemanticBlock]] = []
    current_partition: List[SemanticBlock] = []
    current_size = 0
    for block in blocks:
        if current_size > 0 and current_size + block.char_count > max_chunk_size:
            partitions.append(current_partition)
            current_partition = []
            current_size = 0
        
        current_partition.append(block)
        current_size += block.char_count
        
        if current_size >= target_chunk_size:
            partitions.append(current_partition)
            current_partition = []
            current_size = 0
    
    if current_partition:
        partitions.append(current_partition)
    return partitions

def group_blocks_into_chunks(blocks: List[SemanticBlock], max_chunk_size: int, target_chunk_size: int) -> List[Chunk]:
    """Groups semantic blocks into balanced chunks using dynamic programming."""
    if not blocks: return []

    preprocessed_blocks = [sub_block for block in blocks for sub_block in _split_oversized_block(block, max_chunk_size)]

    n = len(preprocessed_blocks)
    if n == 0: return []

    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    split_points = [-1] * (n + 1)

    for i in range(1, n + 1):
        current_size = 0
        for j in range(i - 1, -1, -1):
            block_size = sum(len(b.content) + (len(b.joiner_to_next) if k < i - 1 else 0) for k, b in enumerate(preprocessed_blocks[j:i]))
            if block_size > max_chunk_size: break
            
            cost = (block_size - target_chunk_size) ** 2
            if dp[j] != float('inf') and dp[j] + cost < dp[i]:
                dp[i] = dp[j] + cost
                split_points[i] = j

    if dp[n] == float('inf'):
        print("Warning: Could not create a perfect partition. The final chunk may be undersized.")
        best_prefix_cost = float('inf')
        best_split_point = -1
        for j in range(n - 1, -1, -1):
            if dp[j] == float('inf'): continue
            remainder_size = sum(b.char_count for b in preprocessed_blocks[j:n])
            if remainder_size > max_chunk_size: continue
            current_total_cost = dp[j] + (remainder_size - target_chunk_size) ** 2
            if current_total_cost < best_prefix_cost:
                best_prefix_cost = current_total_cost
                best_split_point = j
        if best_split_point != -1:
            dp[n] = best_prefix_cost
            split_points[n] = best_split_point

    partitions = []
    if dp[n] != float('inf'):
        idx = n
        while idx > 0:
            prev_idx = split_points[idx]
            if prev_idx == -1:
                partitions = []
                break
            partitions.append(preprocessed_blocks[prev_idx:idx])
            idx = prev_idx
        partitions.reverse()
    
    if not partitions:
        print("Warning: Optimal partitioning failed. Using greedy fallback.")
        partitions = _greedy_fallback(preprocessed_blocks, target_chunk_size, max_chunk_size)
        
    return _create_final_chunks(partitions, blocks)


