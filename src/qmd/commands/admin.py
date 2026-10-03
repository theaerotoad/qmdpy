import os
import sys
import subprocess
from pathlib import Path
from qmd.store import Store
from qmd.formatters.cli import format_report_cli
from qmd.formatters.json import format_report_json, format_report_csv
from qmd.formatters.xml import format_report_xml
from qmd.formatters.core import set_plain_mode
from qmd.formatters.colors import Colors

from .helpers import _is_xml_output

def handle_update(args, store: Store):
    if getattr(args, "verbose", False):
        os.environ["QMD_VERBOSE"] = "1"

    config = store.config
    if getattr(config, "is_federated", False):
        print(f"{Colors.RED}Error: Updating/indexing is disabled in federated include mode. Update individual collection configurations directly.{Colors.RESET}")
        sys.exit(1)
        
    if getattr(args, "build_ann", False):
        store.build_usearch_index()
        return

    # 1. Update existing collections
    for name, coll_cfg in config.collections.items():
        if args.collection:
            c_filter = args.collection.lower()
            if name.lower() != c_filter and c_filter not in name.lower():
                continue

        if args.pull:
            repo_path = Path(coll_cfg.path).expanduser().resolve()
            if (repo_path / ".git").exists():
                print(f"Pulling latest changes for {name}...")
                try:
                    subprocess.run(
                        ["git", "pull"], 
                        cwd=repo_path, 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    print(f"{Colors.GREEN}✓ Git pull successful.{Colors.RESET}")
                except subprocess.CalledProcessError as e:
                    print(f"{Colors.RED}✗ Git pull failed for {name}: {e.stderr.strip()}{Colors.RESET}")
            else:
                print(f"Skipping git pull: {name} is not a git repository.")
        
        store.index_collection(name, coll_cfg, force=args.force, verbose=getattr(args, "verbose", False), quick=getattr(args, "quick", False))

    # 2. Prune removed collections
    active_collections = list(config.collections.keys())
    store.prune_orphaned_collections(active_collections)

    # 3. Automatically build / update the HNSW ANN index
    if not getattr(args, "no_ann", False):
        store.build_usearch_index()

    # 4. Report indexing errors if any remain
    errors = store.get_indexing_errors(collection=args.collection)
    if errors:
        print(f"\n{Colors.YELLOW}Notice: {len(errors)} file(s) have unresolved indexing errors/degradations.{Colors.RESET}")
        for err in errors[:5]:
            print(f"  • [{err['collection']}] {err['path']} - {err['error_type']}: {err['error_message']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more.")
    else:
        print(f"{Colors.GREEN}✓ All files indexed cleanly with zero errors.{Colors.RESET}")

def handle_report(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    # 1. Build Configuration Map from YAML or CLI arguments
    report_cfg = {}
    if getattr(args, "yaml", None):
        import yaml
        try:
            with open(args.yaml, 'r', encoding='utf-8') as f:
                report_cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"{Colors.RED}Error loading YAML file: {e}{Colors.RESET}")
            sys.exit(1)
    else:
        fields = []
        sort_rules = []
        if getattr(args, "fields", None):
            raw_fields = [f.strip() for f in args.fields.split(',') if f.strip()]
            for rf in raw_fields:
                if rf.endswith('+'):
                    sort_rules.append({"field": rf[:-1], "order": "asc"})
                    fields.append(rf[:-1])
                elif rf.endswith('-'):
                    sort_rules.append({"field": rf[:-1], "order": "desc"})
                    fields.append(rf[:-1])
                else:
                    fields.append(rf)
        if fields:
            report_cfg["fields"] = fields
        if sort_rules:
            report_cfg["sort"] = sort_rules

    # Merge CLI limits and targets into config so YAML can be overridden or augmented by CLI
    if "target" not in report_cfg:
        report_cfg["target"] = {}
    
    collection = getattr(args, "collection", None)
    if collection: report_cfg["target"]["collection"] = collection
    
    path = getattr(args, "path", None)
    if path: report_cfg["target"]["path"] = path
    
    title = getattr(args, "title", None)
    if title: report_cfg["target"]["title"] = title

    limit = getattr(args, "limit", 100)

    # Execute custom report compilation
    results = store.build_custom_report(report_cfg=report_cfg, limit=limit)

    if getattr(args, "json", False):
        format_report_json(results)
    elif getattr(args, "csv", False):
        format_report_csv(results)
    elif is_xml:
        format_report_xml(results)
    else:
        format_report_cli(results)

def handle_analyze(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)

    limit = getattr(args, "limit", 100)
    time_limit = getattr(args, "time_limit", None)
    force = getattr(args, "force", False)
    outdated = getattr(args, "outdated", False)
    verbose = getattr(args, "verbose", False)
    
    collection = getattr(args, "collection", None)
    path = getattr(args, "path", None)
    title = getattr(args, "title", None)

    results = store.analyze_target(
        collection=collection,
        path=path,
        title=title,
        limit=limit,
        time_limit=time_limit,
        force=force,
        outdated=outdated,
        verbose=verbose
    )

    if getattr(args, "json", False):
        import json
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print(f"{Colors.YELLOW}No documents found to analyze or all matched documents are already analyzed.{Colors.RESET}")
            return
        print(f"\n{Colors.CYAN}--- Document Analysis Results ---{Colors.RESET}")
        format_report_cli(results)