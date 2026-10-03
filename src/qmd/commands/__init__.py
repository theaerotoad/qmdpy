from .helpers import (
    _clean_header_part,
    _normalize_str,
    _is_xml_output,
    merge_overlapping_snippets,
    group_results_by_doc
)
from .search import handle_discover, handle_map, handle_search
from .read import handle_extract, handle_outline, handle_chunk
from .meta import handle_collection_tree, handle_guide, handle_collections_list
from .admin import handle_update, handle_report, handle_analyze