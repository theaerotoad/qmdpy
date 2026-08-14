import json
import pytest
from pathlib import Path
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.formatting import format_collection_tree_cli, format_collection_tree_xml, set_plain_mode
from qmd.main import main


def seed_documents(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM content")
    docs = [
        ("code", "docs/python/guide.md", "Python Guide", "hash1"),
        ("code", "docs/python/advanced.md", "Advanced Python", "hash2"),
        ("code", "docs/setup.md", "Setup Guide", "hash3"),
        ("code", "src/core/app.py", "App Entry", "hash4"),
        ("code", "README.md", "Readme", "hash5"),
        ("notes", "journal/2026-08.md", "August Journal", "hash6"),
        ("notes", "todo.md", "Todo List", "hash7"),
    ]
    for coll, path, title, h in docs:
        cursor.execute(
            "INSERT INTO content (hash, body, created_at) VALUES (?, ?, '2026-08-14T00:00:00')",
            (h, b"dummy content")
        )
        cursor.execute(
            "INSERT INTO documents (collection, path, title, hash, modified_at) VALUES (?, ?, ?, ?, '2026-08-14T00:00:00')",
            (coll, path, title, h)
        )
    conn.commit()


def test_get_collection_tree_structure(db_conn, temp_db_path, tmp_path):
    seed_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    # Test single collection lookup
    tree_code = store.get_collection_tree("code")
    assert tree_code is not None
    assert tree_code["collection"] == "code"
    root = tree_code["tree"]
    assert root["name"] == "code"
    assert root["type"] == "directory"

    child_names = [c["name"] for c in root["children"]]
    assert "docs" in child_names
    assert "src" in child_names
    assert "README.md" in child_names

    # Test all collections lookup
    all_trees = store.get_collection_tree()
    assert isinstance(all_trees, list)
    assert len(all_trees) == 2
    coll_names = [t["collection"] for t in all_trees]
    assert "code" in coll_names
    assert "notes" in coll_names

    # Test non-existent collection
    assert store.get_collection_tree("unknown_coll") is None


def test_collection_tree_max_depth(db_conn, temp_db_path, tmp_path):
    seed_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    # Depth 1: Should only show top-level files/dirs in root
    tree_d1 = store.get_collection_tree("code", max_depth=1)
    root = tree_d1["tree"]
    for child in root["children"]:
        if child["type"] == "directory":
            assert child["children"] == []


def test_collection_tree_formatting(capsys):
    set_plain_mode(True)
    sample_tree = {
        "collection": "code",
        "tree": {
            "name": "code",
            "type": "directory",
            "children": [
                {
                    "name": "docs",
                    "type": "directory",
                    "children": [
                        {
                            "name": "guide.md",
                            "type": "file",
                            "title": "Python Guide",
                            "path": "docs/guide.md",
                            "doc_id": 1
                        }
                    ]
                },
                {
                    "name": "README.md",
                    "type": "file",
                    "title": "Readme",
                    "path": "README.md",
                    "doc_id": 2
                }
            ]
        }
    }

    # Test ASCII CLI Output
    format_collection_tree_cli(sample_tree)
    out = capsys.readouterr().out
    assert "code/" in out
    assert "docs/" in out
    assert "guide.md" in out
    assert "README.md" in out

    # Test XML Output
    format_collection_tree_xml(sample_tree)
    xml_out = capsys.readouterr().out
    assert '<collection_tree collection="code">' in xml_out
    assert '<directory name="docs">' in xml_out
    assert '<file name="guide.md"' in xml_out
    assert 'path="docs/guide.md"' in xml_out
    assert '</collection_tree>' in xml_out


def test_collection_tree_cli_commands(monkeypatch, capsys, db_conn, temp_db_path, tmp_path):
    seed_documents(db_conn)
    config_file = tmp_path / "config.yml"
    config_file.write_text(f"""
db_path: "{temp_db_path}"
collections:
  code:
    path: "{tmp_path}"
""")

    monkeypatch.setattr("sys.argv", ["qmd", "-C", str(config_file), "collection", "tree", "code", "--json"])
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["collection"] == "code"

    # Test XML output flag
    monkeypatch.setattr("sys.argv", ["qmd", "-C", str(config_file), "collection", "tree", "code", "--xml"])
    main()
    out_xml = capsys.readouterr().out
    assert "<collection_tree collection=\"code\">" in out_xml

    # Test LLM alias flag
    monkeypatch.setattr("sys.argv", ["qmd", "-C", str(config_file), "collection", "tree", "code", "--llm"])
    main()
    out_llm = capsys.readouterr().out
    assert "<collection_tree collection=\"code\">" in out_llm


def test_api_collections_tree(monkeypatch, tmp_path, db_conn, temp_db_path):
    from qmd.web import app
    seed_documents(db_conn)

    config_file = tmp_path / "config.yml"
    config_file.write_text(f"""
db_path: "{temp_db_path}"
collections:
  code:
    path: "{tmp_path}"
  notes:
    path: "{tmp_path}"
""")

    app.config['CONFIG_PATH'] = str(config_file)
    app.config['TESTING'] = True

    with app.test_client() as client:
        # All collections tree
        res = client.get('/api/collections/tree')
        assert res.status_code == 200
        data = res.get_json()
        assert isinstance(data, list)
        assert len(data) == 2

        # Specific collection tree
        res_code = client.get('/api/collections/tree?collection=code')
        assert res_code.status_code == 200
        data_code = res_code.get_json()
        assert data_code["collection"] == "code"

        # Not found collection
        res_404 = client.get('/api/collections/tree?collection=nonexistent')
        assert res_404.status_code == 404
