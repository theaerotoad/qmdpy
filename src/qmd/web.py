import os
import json
import time
import secrets
from flask import Flask, request, jsonify, render_template, g

from qmd.config import load_config
from qmd.store import Store
from qmd.main import group_results_by_doc
from qmd.utils import decompress_text, redact_pii, parse_query_directives
from qmd.db import get_seen_chunks_for_session, record_session_event, record_session_results, get_db_meta

app = Flask(__name__)

def get_config():
    if 'CONFIG_PATH' in app.config and app.config.get('_current_config_path') != app.config['CONFIG_PATH']:
        app.config['config'] = load_config(app.config['CONFIG_PATH'])
        app.config['_current_config_path'] = app.config['CONFIG_PATH']
    elif 'config' not in app.config:
        app.config['config'] = load_config(app.config.get('CONFIG_PATH'))
    return app.config['config']

def get_store():
    """Get a thread-local instance of the Store to avoid SQLite threading errors."""
    if 'store' not in g:
        g.store = Store(get_config())
    return g.store

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        t0 = time.time()
        data = request.json or {}
        raw_query = data.get('query', '')
        clean_query, directives = parse_query_directives(raw_query)
        query = clean_query if clean_query else raw_query

        limit = directives.get('limit') if directives.get('limit') is not None else int(data.get('limit', 10))
        doc_view = data.get('doc', False)
        session_id = data.get('session_id') or secrets.token_hex(4)

        exclude_seen = directives.get('exclude_seen') if directives.get('exclude_seen') is not None else data.get('exclude_seen', False)
        rerank = directives.get('rerank') if directives.get('rerank') is not None else data.get('rerank', False)
        should_redact = directives.get('redact_pii') if directives.get('redact_pii') is not None else (data.get('redact_pii', False) or data.get('redact', False))

        collection = directives.get('collection') or data.get('collection') or None
        lexical_query = directives.get('lex') or data.get('lex') or None
        title = directives.get('title') or data.get('title') or None

        paths_input = data.get('paths') if data.get('paths') is not None else data.get('path')
        if directives.get('path'):
            if paths_input:
                if isinstance(paths_input, list):
                    if directives['path'] not in paths_input:
                        paths_input = paths_input + [directives['path']]
                else:
                    paths_input = [paths_input, directives['path']]
            else:
                paths_input = directives['path']

        store = get_store()
        exclude_seen_set = get_seen_chunks_for_session(store.history_conn, session_id) if exclude_seen else set()

        results = store.hybrid_search(
            query=query,
            limit=limit,
            rerank=rerank,
            collection=collection,
            lexical_query=lexical_query,
            title=title,
            path=paths_input if paths_input else None,
            exclude_seen_set=exclude_seen_set
        )

        if should_redact:
            for r in results:
                r.text = redact_pii(r.text)
                if hasattr(r, 'title') and r.title:
                    r.title = redact_pii(r.title)

        event_type = "doc_view" if doc_view else "search"
        db_last_updated = get_db_meta(store.conn, "last_updated")
        event_id = record_session_event(
            store.history_conn,
            session_id,
            event_type,
            query,
            lexical_query,
            str(store.config.db_path),
            db_last_updated
        )

        excluded_count = store.last_exclusion_stats.get("excluded_chunks", 0)

        if doc_view:
            grouped = group_results_by_doc(results)[:limit]
            shown_chunks = []
            for doc in grouped:
                for c in doc.get("chunks", []):
                    shown_chunks.append({
                        "collection": doc.get("collection", ""),
                        "path": doc.get("path", ""),
                        "seq_id": c.get("seq_id", 0),
                        "rank": c.get("rank", 0),
                        "score": c.get("score", 0.0)
                    })
            record_session_results(store.history_conn, session_id, event_id, shown_chunks)
            time_taken = round(time.time() - t0, 3)
            return jsonify({
                "results": grouped,
                "type": "doc",
                "session_id": session_id,
                "excluded_count": excluded_count,
                "time_taken": time_taken
            })
        else:
            record_session_results(store.history_conn, session_id, event_id, results)
            out = []
            for r in results:
                out.append({
                    "path": r.path,
                    "title": r.title,
                    "text": r.text,
                    "score": r.score,
                    "source": r.source,
                    "collection": r.collection,
                    "seq_id": r.seq_id,
                    "headers": getattr(r, "headers", ""),
                    "rank": getattr(r, "rank", None)
                })
            time_taken = round(time.time() - t0, 3)
            return jsonify({
                "results": out,
                "type": "chunk",
                "session_id": session_id,
                "excluded_count": excluded_count,
                "time_taken": time_taken
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/outline', methods=['GET'])
def get_outline():
    collection = request.args.get('collection')
    path = request.args.get('path')
    if not path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    outline = get_store().get_document_outline(collection=collection, path=path)
    if outline:
        return jsonify(outline)
    return jsonify({"error": "Document not found"}), 404

@app.route('/api/chunk', methods=['GET'])
def get_chunk():
    collection = request.args.get('collection')
    path = request.args.get('path')
    seq_id_arg = request.args.get('seq_id')
    rowid_arg = request.args.get('rowid')
    window = int(request.args.get('window', 0))

    store = get_store()
    results = []

    if rowid_arg is not None:
        try:
            rowid = int(rowid_arg)
            results = store.get_chunk_by_id(rowid, window=window)
        except ValueError:
            return jsonify({"error": "Invalid rowid"}), 400
    elif path is not None:
        seq_id = int(seq_id_arg) if seq_id_arg is not None else 0
        results = store.get_chunk_by_seq(collection=collection, path=path, seq_id=seq_id, window=window)
    else:
        return jsonify({"error": "Must provide 'path' with 'seq_id' or 'rowid'"}), 400

    out = []
    for r in results:
        out.append({
            "path": r.path,
            "title": r.title,
            "text": r.text,
            "collection": r.collection,
            "seq_id": r.seq_id,
            "headers": getattr(r, "headers", "")
        })
    return jsonify({"chunks": out, "total": len(out)})

@app.route('/api/document', methods=['GET'])
def get_document():
    collection = request.args.get('collection')
    path = request.args.get('path')
    should_redact = request.args.get('redact_pii') == 'true' or request.args.get('redact') == 'true'
    if not collection or not path:
        return jsonify({"error": "Missing params"}), 400

    cursor = get_store().conn.cursor()
    cursor.execute('''
        SELECT c.body, d.title
        FROM documents d
        JOIN content c ON d.hash = c.hash
        WHERE d.collection = ? AND d.path = ?
    ''', (collection, path))
    row = cursor.fetchone()

    if row:
        title = row[1]
        content = decompress_text(row[0])
        if should_redact:
            title = redact_pii(title)
            content = redact_pii(content)
        return jsonify({"title": title, "content": content})
    return jsonify({"error": "Not found"}), 404

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    try:
        store = get_store()
        seen = get_seen_chunks_for_session(store.history_conn, session_id)
        cursor = store.history_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM session_events WHERE session_id = ?", (session_id,))
        event_row = cursor.fetchone()
        event_count = event_row[0] if event_row else 0
        return jsonify({
            "session_id": session_id,
            "seen_chunks_count": len(seen),
            "events_count": event_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collections', methods=['GET'])
def collections():
    cfg = get_config()
    store = get_store()
    colls = []
    for k, v in cfg.collections.items():
        doc_count = 0
        try:
            cursor = store.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection = ?", (k,))
            row = cursor.fetchone()
            if row:
                doc_count = row[0]
        except Exception:
            pass
        colls.append({"name": k, "path": str(v.path), "doc_count": doc_count})
    return jsonify(colls)

@app.route('/api/grep', methods=['POST'])
def grep():
    try:
        t0 = time.time()
        data = request.json or {}
        raw_pattern = data.get('pattern', '')
        clean_pattern, directives = parse_query_directives(raw_pattern)
        pattern = clean_pattern if clean_pattern else raw_pattern
        if not pattern:
            return jsonify({"error": "Missing 'pattern' parameter"}), 400

        is_regex = directives.get('regex') if directives.get('regex') is not None else bool(data.get('regex', False))
        case_sensitive = directives.get('case_sensitive') if directives.get('case_sensitive') is not None else bool(data.get('case_sensitive', False))
        collection = directives.get('collection') or data.get('collection') or None
        paths_input = data.get('paths') if data.get('paths') is not None else data.get('path')
        if directives.get('path'):
            if paths_input:
                if isinstance(paths_input, list):
                    if directives['path'] not in paths_input:
                        paths_input = paths_input + [directives['path']]
                else:
                    paths_input = [paths_input, directives['path']]
            else:
                paths_input = directives['path']
        limit = directives.get('limit') if directives.get('limit') is not None else int(data.get('limit', 50))

        store = get_store()
        try:
            results = store.grep_search(
                pattern=pattern,
                is_regex=is_regex,
                case_sensitive=case_sensitive,
                collection=collection,
                path=paths_input if paths_input else None,
                limit=limit
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        time_taken = round(time.time() - t0, 3)
        return jsonify({
            "results": results,
            "total_matches": len(results),
            "pattern": pattern,
            "regex": is_regex,
            "case_sensitive": case_sensitive,
            "time_taken": time_taken
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collections/tree', methods=['GET'])
def get_collections_tree():
    collection = request.args.get('collection')
    depth_str = request.args.get('depth')
    pattern = request.args.get('pattern') or None
    is_regex = request.args.get('regex') == 'true'
    case_sensitive = request.args.get('case_sensitive') == 'true'

    depth = None
    if depth_str is not None:
        try:
            depth = int(depth_str)
        except ValueError:
            return jsonify({"error": "Invalid depth parameter"}), 400

    try:
        tree_data = get_store().get_collection_tree(
            collection=collection,
            max_depth=depth,
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if collection and tree_data is None:
        return jsonify({"error": f"Collection '{collection}' not found"}), 404
    return jsonify(tree_data if tree_data is not None else [])

@app.route('/api/update', methods=['POST'])
def update():
    data = request.json
    collection = data.get('collection')
    force = data.get('force', False)
    cfg = get_config()
    
    if collection and collection in cfg.collections:
        try:
            get_store().index_collection(collection, cfg.collections[collection], force=force)
            return jsonify({"status": "success", "message": f"Successfully updated collection: {collection}"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    return jsonify({"status": "error", "message": "Collection not found"}), 404

def start_server(port=5000, config_path=None):
    if config_path:
        app.config['CONFIG_PATH'] = config_path
        app.config['config'] = load_config(config_path)
    print(f"Starting QMD Web UI at http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)