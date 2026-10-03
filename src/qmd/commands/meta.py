import sys
from typing import Optional
from qmd.store import Store
from qmd.formatters.cli import format_collection_tree_cli
from qmd.formatters.xml import format_collection_tree_xml
from qmd.formatters.core import set_plain_mode, strip_ansi
from qmd.formatters.colors import Colors

from .helpers import _is_xml_output

def handle_collection_tree(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    coll_name = getattr(args, "name", None) or getattr(args, "collection", None)
    pattern = getattr(args, "pattern", None)
    is_regex = getattr(args, "regex", False)
    case_sensitive = getattr(args, "case_sensitive", False) and not getattr(args, "ignore_case", False)

    try:
        tree_data = store.get_collection_tree(
            collection=coll_name,
            max_depth=getattr(args, "depth", None),
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive
        )
    except ValueError as e:
        if args.json:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

    if coll_name and tree_data is None:
        if args.json:
            import json
            print(json.dumps({"error": f"Collection '{coll_name}' not found"}, indent=2))
        else:
            print(f"{Colors.RED}Error: Collection '{coll_name}' not found.{Colors.RESET}")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(tree_data, indent=2))
    elif is_xml:
        format_collection_tree_xml(tree_data)
    else:
        format_collection_tree_cli(tree_data)

def handle_guide(args, store: Optional[Store] = None):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    if is_xml:
        guide_xml = """<qmd_guide>
  <overview>QMD is a local search, retrieval, and document inspection engine designed for LLM agents. Formulate queries as natural language questions rather than keyword searches.</overview>
  <batch_execution_format>
When requested to perform research or retrieve data using QMD, emit 1 to 5 commands wrapped in a <qmd_commands> XML container. Frame queries as clear natural language questions:

<qmd_commands>
  qmd discover "what are the main principles of orbital mechanics?"
  qmd search "how do orbital transfer maneuvers work?"
  qmd outline "coll:path.md"
  qmd read "coll:path.md:10-15"
</qmd_commands>
  </batch_execution_format>
  <workflow>
    <step num="1" name="Discovery">Use `qmd discover "natural language question?"` for top-level SERP results (1 hit per document), or `qmd map "question?"` to view a semantic density tree of results across folders.</step>
    <step num="2" name="Search">Use `qmd search "natural language question?"` for comprehensive document-grouped results.</step>
    <step num="3" name="Orient">Use `qmd outline "<target>"` to inspect heading hierarchies and chunk sequence spans.</step>
    <step num="4" name="Read">Use `qmd read "<target>"` to fetch chunks/ranges, or execute exact `read="..."` / `expand="..."` attributes from search results.</step>
  </workflow>
  <shorthand_targets>
    <target syntax="coll:path:seq_range">e.g. `Books:history.epub:10-15` or `qmd://Books/history.epub:10-15`</target>
    <target syntax="coll:path">e.g. `Books:history.epub` or `qmd://Books/history.epub`</target>
    <target syntax="path:seq_range">e.g. `notes.md:0-3`</target>
    <target syntax="row_ids">e.g. `10-15` or `22,40,25-27`</target>
  </shorthand_targets>
  <agent_tips>
    <tip>Natural language questions: formulate search queries as natural language questions rather than keyword lists for optimal semantic retrieval.</tip>
    <tip>Triage first: `qmd discover "question?"` gives a fast 1-hit-per-document overview before deep reading.</tip>
    <tip>Batch execution: emit 1 to 5 sequential commands in a <qmd_commands> block for unified batch processing.</tip>
    <tip>Agent hypermedia: copy-paste the `read="..."`, `outline="..."`, and `search="..."` attributes directly into CLI calls.</tip>
    <tip>Safety cap: large reads truncate at max_chunks (default 30) and provide a `resume="..."` command for the next slice.</tip>
  </agent_tips>
</qmd_guide>"""
        print(guide_xml)
        return guide_xml
    else:
        guide_md = f"""{Colors.BOLD}QMD LLM Agent Research & Inspection Guide{Colors.RESET}

{Colors.CYAN}## Workflow & Decision Matrix{Colors.RESET}
1. {Colors.BOLD}Triage & Discovery:{Colors.RESET}
   - `qmd discover "question?"` - Single top hit per document SERP view (use natural language questions)
   - `qmd map "question?"` - Semantic directory tree mapping (density clustering)
   - `qmd collections` - List indexed collections and paths
   - `qmd tree [collection] [-p pattern]` - Explore document directory trees

2. {Colors.BOLD}Retrieval & Search:{Colors.RESET}
   - `qmd search "question?"` - Document-ordered search results (natural language question)
   - `qmd search "question?" --deep` - Deep search: doc-grouped + LLM reranked
   - `qmd search "question?" --session <id>` - Session search (auto-deduplicates seen chunks)
   - `qmd search "question?" --broad` - Broad hierarchical search (wide-to-narrow)

3. {Colors.BOLD}Document Structure & Orientation:{Colors.RESET}
   - `qmd outline "<target>"` - Table of contents with chunk sequence mappings

4. {Colors.BOLD}Targeted Reading:{Colors.RESET}
   - `qmd read "<target>"` - Read specific chunks, ranges, or surrounding context
   - Examples: `qmd read 'Books:doc.epub:10-15'`, `qmd read 'qmd://Books/doc.epub:3'`

{Colors.CYAN}## Target Shorthand Syntax{Colors.RESET}
- Full URI: `qmd://<collection>/<path>[:<seq>]`
- Shorthand: `<collection>:<path>[:<seq>]`
- Relative: `<path>[:<seq>]`
- Chunk Row IDs: `10-15` or `22,40,25-27`

{Colors.CYAN}## Query Formulation & Agent Best Practices{Colors.RESET}
- Frame queries as clear natural language questions rather than keyword lists for optimal semantic retrieval.
- Use `qmd discover` to scan across multiple documents without flooding context.
- Pass `--session <id>` when conducting multi-turn research to avoid ingesting duplicate context.
- Follow actionable XML attributes (`read="..."`, `outline="..."`, `search="..."`, `resume="..."`).
- Large reads truncate at 30 chunks with a resume hint to protect context windows."""
        if getattr(args, "plain", False):
            guide_md = strip_ansi(guide_md)
        res = guide_md.strip()
        print(res)
        return res

def handle_collections_list(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)
    if getattr(args, "json", False):
        import json
        data = {name: {"path": cfg.path, "glob": cfg.glob} for name, cfg in store.config.collections.items()}
        print(json.dumps(data, indent=2))
    else:
        for name, cfg in store.config.collections.items():
            print(f"{Colors.GREEN}{name}{Colors.RESET}: {cfg.path} ({cfg.glob})")