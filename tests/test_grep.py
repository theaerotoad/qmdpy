import json
import pytest
from pathlib import Path
from qmd.config import Config, load_config
from qmd.store import Store
from qmd.utils import compress_text
from qmd.formatting import format_grep_cli, format_grep_xml, format_grep_json, set_plain_mode
from qmd.main import main


def seed_grep_documents(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM content")
    docs = [
        (
            "code",
            "src/utils.py",
            "Utils Module",
            "hash_u",
            "import os\nimport re\n\ndef _sanitize_term(t: str):\n    return t.strip()\n\ndef extract_terms():\n    pass\n"
        ),
        (
            "code",
            "src/models.py",
            "Data Models",
            "hash_m",
            "class UserModel:\n    def _init_db(self):\n        pass\n    def get_user(self):\n        return 'admin'\n"
        ),
        (
            "notes",
            "guide.md",
            "User Guide",
            "hash_g",
            "# User Guide\n\nTo configure, import sqlite3 database adapter.\nAlways sanitize inputs.\n"
        ),
    ]
    for coll, path, title, h, body in docs:
        cursor.execute(
            "INSERT INTO content (hash, body, created_at) VALUES (?, ?, '2026-08-14T00:00:00')",
            (h, compress_text(body))
        )
        cursor.execute(
            "INSERT INTO documents (collection, path, title, hash, modified_at) VALUES (?, ?, ?, ?, '2026-08-14T00:00:00')",
            (coll, path, title, h)
        )
    conn.commit()


def test_grep_exact_substring(db_conn, temp_db_path):
    seed_grep_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    results = store.grep_search("sanitize")
    assert len(results) == 2
    paths = [r["path"] for r in results]
    assert "src/utils.py" in paths
    assert "guide.md" in paths

    line_nums = {r["path"]: r["line_number"] for r in results}
    assert line_nums["src/utils.py"] == 4
    assert line_nums["guide.md"] == 4


def test_grep_regex_matching(db_conn, temp_db_path):
    seed_grep_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    results = store.grep_search(r"def _\w+", is_regex=True)
    assert len(results) == 2
    match_texts = [r["match_text"] for r in results]
    assert "def _sanitize_term" in match_texts
    assert "def _init_db" in match_texts


def test_grep_case_sensitivity_and_filters(db_conn, temp_db_path):
    seed_grep_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    # Case sensitive match
    res_lower = store.grep_search("user", case_sensitive=True)
    assert len(res_lower) == 1
    assert res_lower[0]["path"] == "src/models.py"

    res_upper = store.grep_search("User", case_sensitive=True)
    assert len(res_upper) == 2
    assert "src/models.py" in [r["path"] for r in res_upper]
    assert "guide.md" in [r["path"] for r in res_upper]

    # Collection filter
    res_coll = store.grep_search("sanitize", collection="notes")
    assert len(res_coll) == 1
    assert res_coll[0]["collection"] == "notes"

    # Path filter
    res_path = store.grep_search("def", path="models")
    assert len(res_path) == 2
    assert all(r["path"] == "src/models.py" for r in res_path)


def test_grep_invalid_regex(db_conn, temp_db_path):
    seed_grep_documents(db_conn)
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    with pytest.raises(ValueError, match="Invalid regular expression"):
        store.grep_search("[unclosed_bracket", is_regex=True)


def test_grep_formatting_cli_and_xml(capsys):
    set_plain_mode(True)
    sample_results = [
        {
            "collection": "code",
            "path": "src/utils.py",
            "title": "Utils Module",
            "line_number": 4,
            "line_text": "def _sanitize_term(t: str):",
            "match_text": "_sanitize_term"
        },
        {
            "collection": "code",
            "path": "src/utils.py",
            "title": "Utils Module",
            "line_number": 6,
            "line_text": "def extract_terms():",
            "match_text": "terms"
        }
    ]

    # CLI Output
    format_grep_cli(sample_results, pattern="terms")
    out = capsys.readouterr().out
    assert "qmd://code/src/utils.py" in out
    assert "4:" in out
    assert "6:" in out

    # XML Output
    format_grep_xml(sample_results, pattern="terms", is_regex=False, case_sensitive=False)
    xml_out = capsys.readouterr().out
    assert '<grep_results pattern="terms"' in xml_out
    assert '<document uri="qmd://code/src/utils.py"' in xml_out
    assert '<match line="4">def _sanitize_term(t: str):</match>' in xml_out
    assert '</grep_results>' in xml_out


def test_grep_cli_command(monkeypatch, capsys, db_conn, temp_db_path, tmp_path):
    seed_grep_documents(db_conn)
    config_file = tmp_path / "config.yml"
    config_file.write_text(f"""
db_path: "{temp_db_path}"
collections:
  code:
    path: "{tmp_path}"
""")

    # JSON grep
    monkeypatch.setattr("sys.argv", ["qmd", "-C", str(config_file), "grep", "import", "--json"])
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) >= 2

    # XML / LLM grep
    monkeypatch.setattr("sys.argv", ["qmd", "-C", str(config_file), "grep", "import", "--llm"])
    main()
    out_llm = capsys.readouterr().out
    assert '<grep_results pattern="import"' in out_llm


def test_api_grep(tmp_path, db_conn, temp_db_path):
    from qmd.web import app
    seed_grep_documents(db_conn)

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
    app.config['config'] = load_config(config_file)
    app.config['TESTING'] = True

    with app.test_client() as client:
        # Standard Grep
        res = client.post('/api/grep', json={"pattern": "sanitize"})
        assert res.status_code == 200
        data = res.get_json()
        assert data["total_matches"] == 2

        # Regex Grep
        res_regex = client.post('/api/grep', json={"pattern": r"def\s+_\w+", "regex": True})
        assert res_regex.status_code == 200
        data_regex = res_regex.get_json()
        assert data_regex["total_matches"] == 2

        # Invalid Regex 400
        res_err = client.post('/api/grep', json={"pattern": "[invalid", "regex": True})
        assert res_err.status_code == 400

        # Missing Pattern 400
        res_missing = client.post('/api/grep', json={})
        assert res_missing.status_code == 400
