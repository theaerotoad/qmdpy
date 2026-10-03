import argparse
import sys
import os
from typing import Optional, Set

from qmd.config import load_config
from qmd.store import Store
from qmd.formatters.colors import Colors

from qmd.commands import (
    handle_discover, handle_map, handle_search, handle_extract,
    handle_outline, handle_chunk, handle_collection_tree, handle_guide,
    handle_collections_list, handle_update, handle_report, handle_analyze,
    group_results_by_doc, merge_overlapping_snippets
)

class HelpAllAction(argparse.Action):
    root_parser: Optional[argparse.ArgumentParser] = None

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help="Show help for all commands and subcommands and exit"):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        target_parser = HelpAllAction.root_parser or parser
        print(format_help_all(target_parser))
        parser.exit(0)

def format_help_all(parser: argparse.ArgumentParser) -> str:
    """
    Recursively collects and formats help text for the root parser and all subcommands.
    """
    sections = [parser.format_help().rstrip()]

    def _collect_subparsers(p: argparse.ArgumentParser, seen: Set[int]):
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, subp in action.choices.items():
                    if id(subp) in seen:
                        continue
                    seen.add(id(subp))
                    sections.append(subp.format_help().rstrip())
                    _collect_subparsers(subp, seen)

    seen = {id(parser)}
    _collect_subparsers(parser, seen)
    separator = "\n\n" + "=" * 80 + "\n\n"
    return separator.join(sections) + "\n"

def build_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-C", "--config", type=str, default=argparse.SUPPRESS, help="Path to custom YAML configuration file")
    parent_parser.add_argument("--helpall", action=HelpAllAction, help="Show help for all commands and subcommands and exit")

    parser = argparse.ArgumentParser(prog="qmd", description="Query Multiple Documents", parents=[parent_parser])
    HelpAllAction.root_parser = parser
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", aliases=["find", "disc"], help="Discover top document matches (1 top hit per document)", parents=[parent_parser])
    discover_parser.add_argument("query", nargs="+", help="The natural language question or search terms")

    d_filter_group = discover_parser.add_argument_group("Target & Filters")
    d_filter_group.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    d_filter_group.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    d_filter_group.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    d_filter_group.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")

    d_mode_group = discover_parser.add_argument_group("Search Mode & Quality")
    d_mode_group.add_argument("--deep", action="store_true", help="Deep discovery: enable LLM reranking")
    d_mode_group.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    d_mode_group.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    d_mode_group.add_argument("-w", "--w2n", action="store_true", help="Use Wide-to-Narrow hierarchical search")
    d_mode_group.add_argument("--broad", action="store_true", help="Broad search: alias for Wide-to-Narrow hierarchical search (--w2n)")
    d_mode_group.add_argument("--limit", type=int, default=None, help="Number of final documents to show")
    d_mode_group.add_argument("--max-chunks", type=int, default=None, help="Max documents to return (defaults to config max_chunks_per_response or 30)")
    d_mode_group.add_argument("--no-cache", action="store_true", help="Bypass and do not write to search result cache")
    d_mode_group.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    d_mode_group.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    d_mode_group.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")
    d_mode_group.add_argument("--dirlist", action="store_true", help="Include dynamic directory listings in search results")

    d_session_group = discover_parser.add_argument_group("Session & History")
    d_session_group.add_argument("--session", type=str, help="Session ID for tracking history and deduplication")
    d_session_group.add_argument("--exclude-seen", action="store_true", help="Exclude previously seen chunks from the active session")
    d_session_group.add_argument("--include-seen", action="store_true", help="Include previously seen chunks (overrides default session deduplication)")

    d_output_group = discover_parser.add_argument_group("Output & Formatting")
    d_output_group.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    d_output_group.add_argument("--llm", action="store_true", help="Alias for --xml (optimizes output for LLM agent context)")
    d_output_group.add_argument("--json", action="store_true", help="Output results in JSON format")
    d_output_group.add_argument("--redact-pii", "--redact", action="store_true", help="Redact email addresses and phone numbers from search results")
    d_output_group.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    d_output_group.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")

    map_parser = subparsers.add_parser("map", aliases=["m"], help="Semantic directory tree mapping (Wide-to-Narrow density clustering)", parents=[parent_parser])
    map_parser.add_argument("query", nargs="+", help="The natural language question or search terms")

    m_filter_group = map_parser.add_argument_group("Target & Filters")
    m_filter_group.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    m_filter_group.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    m_filter_group.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    m_filter_group.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")

    m_mode_group = map_parser.add_argument_group("Search Mode & Quality")
    m_mode_group.add_argument("--deep", action="store_true", help="Deep map: enable LLM reranking")
    m_mode_group.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    m_mode_group.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    m_mode_group.add_argument("--limit", type=int, default=100, help="Number of final documents to sample for density map (default: 100)")
    m_mode_group.add_argument("--no-cache", action="store_true", help="Bypass and do not write to search result cache")
    m_mode_group.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    m_mode_group.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    m_mode_group.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")
    m_mode_group.add_argument("--dirlist", action="store_true", help="Include dynamic directory listings in search results")

    m_output_group = map_parser.add_argument_group("Output & Formatting")
    m_output_group.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    m_output_group.add_argument("--llm", action="store_true", help="Alias for --xml (optimizes output for LLM agent context)")
    m_output_group.add_argument("--json", action="store_true", help="Output results in JSON format")
    m_output_group.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    m_output_group.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")

    search_parser = subparsers.add_parser("search", aliases=["query", "q"], help="Hybrid vector + lexical search (document-ordered by default)", parents=[parent_parser])
    search_parser.add_argument("query", nargs="+", help="The natural language question or search terms")

    filter_group = search_parser.add_argument_group("Target & Filters")
    filter_group.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    filter_group.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    filter_group.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    filter_group.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")

    mode_group = search_parser.add_argument_group("Search Mode & Quality")
    mode_group.add_argument("--deep", action="store_true", help="Deep search: document grouping with LLM reranking (implies -r)")
    mode_group.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    mode_group.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    mode_group.add_argument("-w", "--w2n", action="store_true", help="Use Wide-to-Narrow hierarchical search")
    mode_group.add_argument("--broad", action="store_true", help="Broad search: alias for Wide-to-Narrow hierarchical search (--w2n)")
    mode_group.add_argument("--limit", type=int, default=None, help="Number of final results to show")
    mode_group.add_argument("--max-chunks", type=int, default=None, help="Max chunks to return (defaults to config max_chunks_per_response or 30)")
    mode_group.add_argument("--no-cache", action="store_true", help="Bypass and do not write to search result cache")
    mode_group.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    mode_group.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    mode_group.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")
    mode_group.add_argument("--dirlist", action="store_true", help="Include dynamic directory listings in search results")

    session_group = search_parser.add_argument_group("Session & History")
    session_group.add_argument("--session", type=str, help="Session ID for tracking history and deduplication")
    session_group.add_argument("--exclude-seen", action="store_true", help="Exclude previously seen chunks from the active session")
    session_group.add_argument("--include-seen", action="store_true", help="Include previously seen chunks (overrides default session deduplication)")

    output_group = search_parser.add_argument_group("Output & Formatting")
    output_group.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    output_group.add_argument("--llm", action="store_true", help="Alias for --xml (optimizes output for LLM agent context)")
    output_group.add_argument("--json", action="store_true", help="Output results in JSON format")
    output_group.add_argument("-d", "--doc", action="store_true", default=True, help="Group results by document (default)")
    output_group.add_argument("--flat", "--chunks", action="store_true", help="Output flat individual chunks instead of document-grouped view")
    output_group.add_argument("--redact-pii", "--redact", action="store_true", help="Redact email addresses and phone numbers from search results")
    output_group.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")

    extract_parser = subparsers.add_parser("extract", help="Smart document extraction (representative sampling) with constraints", parents=[parent_parser])
    extract_parser.add_argument("path", help="Path or relative path to the document")
    extract_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    extract_parser.add_argument("-q", "--queries", type=str, nargs="*", default=[], help="Semantic queries to prioritize chunks")
    extract_parser.add_argument("--max-chunks", type=int, default=30, help="Maximum number of chunks to return")
    extract_parser.add_argument("--head", type=int, default=3, help="Number of introduction chunks to unconditionally include")
    extract_parser.add_argument("--tail", type=int, default=3, help="Number of conclusion chunks to unconditionally include")
    extract_parser.add_argument("--top-k", type=int, default=3, help="Top K chunks to include per semantic query")
    extract_parser.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank query results")
    extract_parser.add_argument("--deep", action="store_true", help="Deep extraction: enable LLM reranking (implies -r)")
    extract_parser.add_argument("--json", action="store_true", help="Output as JSON")
    extract_parser.add_argument("--xml", action="store_true", help="Output as XML for LLM context")
    extract_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    extract_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    outline_parser = subparsers.add_parser("outline", help="Show heading outline and chunk mapping for a document", parents=[parent_parser])
    outline_parser.add_argument("path", help="Path or relative path to the document")
    outline_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    outline_parser.add_argument("-d", "--depth", type=int, default=None, help="Maximum heading depth to display (e.g. 1 for H1 only, 2 for H1-H2)")
    outline_parser.add_argument("-p", "--pattern", type=str, default=None, help="Filter headings by substring pattern")
    outline_parser.add_argument("--json", action="store_true", help="Output outline as JSON")
    outline_parser.add_argument("--xml", action="store_true", help="Output outline as XML for LLM context")
    outline_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    outline_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    read_parser = subparsers.add_parser("read", aliases=["chunk", "get", "view"], help="Fetch specific chunk by rowid, path, or URI with surrounding context window", parents=[parent_parser])
    read_parser.add_argument("target", help="Document path, URI (qmd://...), shorthand (coll:path:seq), OR chunk rowid/ranges")
    read_parser.add_argument("--seq", type=str, default=None, help="Sequence ID or range of chunk(s) (when target is a document path, e.g. 0, 1-5, 2,4,6-8)")
    read_parser.add_argument("-w", "--window", type=int, default=0, help="Number of surrounding chunks to fetch on each side")
    read_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    read_parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to return per read call (defaults to config max_chunks_per_response or 30)")
    read_parser.add_argument("--json", action="store_true", help="Output chunks as JSON")
    read_parser.add_argument("--xml", action="store_true", help="Output chunks as XML for LLM context")
    read_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    read_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    tree_parser = subparsers.add_parser("tree", help="Display hierarchical directory tree of indexed files", parents=[parent_parser])
    tree_parser.add_argument("name", nargs="?", default=None, help="Optional collection name")
    tree_parser.add_argument("-c", "--collection", type=str, default=None, help="Filter by collection name")
    tree_parser.add_argument("-p", "--pattern", type=str, default=None, help="Pattern or substring to filter file paths and titles")
    tree_parser.add_argument("-r", "--regex", action="store_true", help="Treat pattern as a regular expression")
    tree_parser.add_argument("-s", "--case-sensitive", action="store_true", help="Perform case-sensitive matching")
    tree_parser.add_argument("-i", "--ignore-case", action="store_true", help="Perform case-insensitive matching (default)")
    tree_parser.add_argument("--depth", type=int, default=None, help="Maximum directory depth to display")
    tree_parser.add_argument("--json", action="store_true", help="Output directory tree as JSON")
    tree_parser.add_argument("--xml", action="store_true", help="Output directory tree as XML for LLM context")
    tree_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    tree_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    colls_parser = subparsers.add_parser("collections", aliases=["colls"], help="List configured collections and paths", parents=[parent_parser])
    colls_parser.add_argument("--json", action="store_true", help="Output collections list as JSON")
    colls_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    guide_parser = subparsers.add_parser("guide", help="Emit LLM agent research guide and command decision matrix", parents=[parent_parser])
    guide_parser.add_argument("--xml", action="store_true", help="Output guide in XML format")
    guide_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    guide_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    env_xml = os.environ.get("QMD_XML", "").strip().lower() in ("1", "true", "yes", "on")
    env_hide_mcp = os.environ.get("QMD_HIDE_MCP", "").strip().lower() in ("1", "true", "yes", "on")
    hide_advanced = env_xml or env_hide_mcp

    update_help = argparse.SUPPRESS if hide_advanced else "Update the index"
    update_parser = subparsers.add_parser("update", help=update_help, parents=[parent_parser])
    update_parser.add_argument("--pull", action="store_true", help="Run 'git pull' before indexing")
    update_parser.add_argument("-f", "--force", action="store_true", help="Force re-indexing of all files, ignoring hash checks")
    update_parser.add_argument("-q", "--quick", action="store_true", help="Quick update: skip hashing for files with unchanged size and modification time")
    update_parser.add_argument("-c", "--collection", type=str, help="Only update a specific collection")
    update_parser.add_argument("--build-ann", action="store_true", help="Build a usearch HNSW approximate nearest neighbor index from the existing vector table")
    update_parser.add_argument("--no-ann", action="store_true", help="Skip automatic HNSW ANN index build/update")
    update_parser.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info during update")

    analyze_help = argparse.SUPPRESS if hide_advanced else "Run LLM document summarization and metadata extraction"
    analyze_parser = subparsers.add_parser("analyze", aliases=["analysis"], help=analyze_help, parents=[parent_parser])
    analyze_parser.add_argument("-c", "--collection", type=str, help="Filter documents by collection name")
    analyze_parser.add_argument("-p", "--path", type=str, help="Filter documents by path (substring match)")
    analyze_parser.add_argument("-t", "--title", type=str, help="Filter documents by title (substring match)")
    analyze_parser.add_argument("--limit", type=int, default=100, help="Maximum number of documents to analyze in this run")
    analyze_parser.add_argument("--time-limit", type=float, default=None, help="Maximum execution time in hours (e.g., 0.5 for 30 mins)")
    analyze_parser.add_argument("-f", "--force", action="store_true", help="Force re-analysis even if document is already analyzed")
    analyze_parser.add_argument("--outdated", action="store_true", help="Analyze missing documents AND re-analyze those processed with an older prompt version")
    analyze_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    analyze_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output during analysis")

    report_parser = subparsers.add_parser("report", help="Display existing document analysis reports", parents=[parent_parser])
    report_parser.add_argument("-c", "--collection", type=str, help="Filter documents by collection name")
    report_parser.add_argument("-p", "--path", type=str, help="Filter documents by path (substring match)")
    report_parser.add_argument("-t", "--title", type=str, help="Filter documents by title (substring match)")
    report_parser.add_argument("--limit", type=int, default=100, help="Maximum number of reports to display")
    report_parser.add_argument("--fields", type=str, help="Comma-separated fields to display (e.g. 'dates+,altTitle,type-')")
    report_parser.add_argument("-y", "--yaml", type=str, help="Path to a YAML report configuration file")
    report_parser.add_argument("--csv", action="store_true", help="Output results as CSV")
    report_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    report_parser.add_argument("--xml", action="store_true", help="Output results as XML for LLM context")
    report_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    report_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    coll_parser = subparsers.add_parser("collection", help="Manage collections", parents=[parent_parser])
    coll_sub = coll_parser.add_subparsers(dest="subcommand", required=True)
    coll_list = coll_sub.add_parser("list", help="List all configured collections", parents=[parent_parser])

    coll_tree = coll_sub.add_parser("tree", help="Display directory tree of indexed files", parents=[parent_parser])
    coll_tree.add_argument("name", nargs="?", default=None, help="Optional collection name")
    coll_tree.add_argument("-c", "--collection", type=str, default=None, help="Filter by collection name")
    coll_tree.add_argument("-p", "--pattern", type=str, default=None, help="Pattern or substring to filter file paths and titles")
    coll_tree.add_argument("-r", "--regex", action="store_true", help="Treat pattern as a regular expression")
    coll_tree.add_argument("-s", "--case-sensitive", action="store_true", help="Perform case-sensitive matching")
    coll_tree.add_argument("-i", "--ignore-case", action="store_true", help="Perform case-insensitive matching (default)")
    coll_tree.add_argument("--depth", type=int, default=None, help="Maximum directory depth to display")
    coll_tree.add_argument("--json", action="store_true", help="Output directory tree as JSON")
    coll_tree.add_argument("--xml", action="store_true", help="Output directory tree as XML for LLM context")
    coll_tree.add_argument("--llm", action="store_true", help="Alias for --xml")
    coll_tree.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    serve_help = argparse.SUPPRESS if hide_advanced else "Start the web UI"
    serve_parser = subparsers.add_parser("serve", help=serve_help, parents=[parent_parser])
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")

    mcp_help = argparse.SUPPRESS if hide_advanced else "Start the stdio MCP server"
    mcp_parser = subparsers.add_parser("mcp", help=mcp_help, parents=[parent_parser])

    return parser

def execute_command(args, store):
    if args.command in ["discover", "find", "disc"]:
        handle_discover(args, store)
    elif args.command in ["map", "m"]:
        handle_map(args, store)
    elif args.command in ["search", "query", "q"]:
        handle_search(args, store)
    elif args.command == "extract":
        handle_extract(args, store)
    elif args.command == "outline":
        handle_outline(args, store)
    elif args.command in ["read", "chunk", "get", "view"]:
        handle_chunk(args, store)
    elif args.command == "tree":
        handle_collection_tree(args, store)
    elif args.command in ["collections", "colls"]:
        handle_collections_list(args, store)
    elif args.command == "guide":
        handle_guide(args, store)
    elif args.command == "update":
        handle_update(args, store)
    elif args.command in ["analyze", "analysis"]:
        handle_analyze(args, store)
    elif args.command == "report":
        handle_report(args, store)
    elif args.command == "collection":
        if args.subcommand == "list":
            handle_collections_list(args, store)
        elif args.subcommand == "tree":
            handle_collection_tree(args, store)
    elif args.command == "serve":
        from qmd.web import start_server
        start_server(port=args.port, config_path=getattr(args, "config", None))
    elif args.command == "mcp":
        from qmd.mcp_server import run_mcp_server
        run_mcp_server(getattr(args, "config", None))

def main():
    parser = build_parser()
    args = parser.parse_args()
    config_path = getattr(args, "config", None)
    
    if args.command == "mcp":
        from qmd.mcp_server import run_mcp_server
        run_mcp_server(config_path)
        return

    config = load_config(config_path)
    is_write = args.command in ["update", "analyze", "analysis"]
    if is_write and getattr(config, "is_federated", False):
        print(f"{Colors.RED}Error: Modifying databases (update/analyze) is disabled in federated include mode. Run against individual collections directly.{Colors.RESET}")
        sys.exit(1)
    store = Store(config, read_only=not is_write)

    try:
        execute_command(args, store)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()