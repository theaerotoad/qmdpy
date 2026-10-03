from .colors import Colors, c
from .core import (
    strip_ansi, escape_xml_attr, escape_xml_text,
    extract_lexical_terms, highlight_keywords, set_plain_mode
)
from .cli import (
    format_results_cli, format_doc_results_cli, format_discover_cli,
    format_outline_cli, format_chunks_cli, format_report_cli,
    format_map_cli, format_collection_tree_cli, format_extraction_cli
)
from .xml import (
    format_results_xml, format_doc_results_xml, format_discover_xml,
    format_outline_xml, format_chunks_xml, format_report_xml,
    format_map_xml, format_collection_tree_xml, format_extraction_xml,
    MAX_GAP_EXPAND_CHUNKS
)
from .json import (
    format_results_json, format_doc_results_json, format_discover_json,
    format_report_json, format_report_csv, format_collection_tree_json
)