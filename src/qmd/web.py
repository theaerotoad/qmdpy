import os
import json
import secrets
from flask import Flask, request, jsonify, render_template, g

from qmd.config import load_config
from qmd.store import Store
from qmd.main import group_results_by_doc
from qmd.utils import decompress_text, redact_pii
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
        data = request.json or {}
        query = data.get('query', '')
        limit = int(data.get('limit', 10))
        doc_view = data.get('doc', False)
        session_id = data.get('session_id') or secrets.token_hex(4)
        exclude_seen = data.get('exclude_seen', False)

        store = get_store()
        exclude_seen_set = get_seen_chunks_for_session(store.history_conn, session_id) if exclude_seen else set()

        results = store.hybrid_search(
            query=query,
            limit=limit,
            rerank=data.get('rerank', False),
            collection=data.get('collection') if data.get('collection') else None,
            lexical_query=data.get('lex') if data.get('lex') else None,
            title=data.get('title') if data.get('title') else None,
            path=data.get('path') if data.get('path') else None,
            exclude_seen_set=exclude_seen_set
        )

        if data.get('redact_pii', False) or data.get('redact', False):
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
            data.get('lex') if data.get('lex') else None,
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
            return jsonify({
                "results": grouped,
                "type": "doc",
                "session_id": session_id,
                "excluded_count": excluded_count
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
            return jsonify({
                "results": out,
                "type": "chunk",
                "session_id": session_id,
                "excluded_count": excluded_count
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

@app.route('/api/collections', methods=['GET'])
def collections():
    cfg = get_config()
    colls = [{"name": k, "path": v.path} for k, v in cfg.collections.items()]
    return jsonify(colls)

@app.route('/api/grep', methods=['POST'])
def grep():
    try:
        data = request.json or {}
        pattern = data.get('pattern', '')
        if not pattern:
            return jsonify({"error": "Missing 'pattern' parameter"}), 400

        is_regex = bool(data.get('regex', False))
        case_sensitive = bool(data.get('case_sensitive', False))
        collection = data.get('collection') or None
        path = data.get('path') or None
        limit = int(data.get('limit', 50))

        store = get_store()
        try:
            results = store.grep_search(
                pattern=pattern,
                is_regex=is_regex,
                case_sensitive=case_sensitive,
                collection=collection,
                path=path,
                limit=limit
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        return jsonify({
            "results": results,
            "total_matches": len(results),
            "pattern": pattern,
            "regex": is_regex,
            "case_sensitive": case_sensitive
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collections/tree', methods=['GET'])
def get_collections_tree():
    collection = request.args.get('collection')
    depth_str = request.args.get('depth')
    depth = None
    if depth_str is not None:
        try:
            depth = int(depth_str)
        except ValueError:
            return jsonify({"error": "Invalid depth parameter"}), 400

    tree_data = get_store().get_collection_tree(collection=collection, max_depth=depth)
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