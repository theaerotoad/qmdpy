# Web Setup GUI Architecture Plan

## 1. CLI Integration (`src/qmd/main.py`)
- Add `--setup` (action='store_true') to the `serve` subcommand.
- Add `--yamldir` (type=str, default=".") to the `serve` subcommand.

## 2. Server Mode (`src/qmd/web.py`)
Intercept the setup flag in `start_server()`. If `setup=True`:
- Skip initializing `Store` or DB connections.
- Serve a distinct set of routes:
  - `GET /`: Renders `setup.html`.
  - `GET /api/yamls`: Returns a list of available configs in the target directory.
  - `GET /api/yamls/<file>`: Returns parsed JSON of a specific config.
  - `POST /api/yamls/<file>`: Backs up the old file to `<file>.bak` and saves the new YAML payload.

## 3. Frontend UI (`src/qmd/templates/setup.html`)
- **Sidebar**: List of `.yml`/`.yaml` files found in the directory.
- **Main Pane**: Form structured by config categories:
  - Core Settings (DB paths, target sizes).
  - LLM / Embedding Models.
  - Search / Retreival Limits.
  - Collections Array (dynamic addition/removal of `CollectionConfig` dictionaries).
- **Actions**: "Save Configuration" (triggers the backup & overwrite).

## 4. Dependencies
- Relies on `PyYAML` (already in `requirements.txt`).
- Assumes existing CSS classes from `app.css` are reusable for form inputs and layouts.
