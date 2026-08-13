import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Union

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "qmd" / "index.yml"

@dataclass
class CollectionConfig:
    path: str
    glob: str = "**/*.md"
    contexts: Optional[Dict[str, str]] = None
    file_extensions: Optional[List[str]] = None
    convert_non_md: bool = True

@dataclass
class Config:
    collections: Dict[str, CollectionConfig] = field(default_factory=dict)
    db_path: Optional[str] = None 
    history_db_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Core LLM Settings
    llm_url: str = "http://127.0.0.1:8888"
    api_key: Optional[str] = None
    embed_url: Optional[str] = None
    rerank_url: Optional[str] = None
    embed_api_key: Optional[str] = None
    rerank_api_key: Optional[str] = None
    request_timeout: float = 120.0
    embed_batch_size: int = 16
    
    # Model Configurations
    embed_model: str = "EmbeddingGemma 300m"
    rerank_model: str = "Qwen Rerank 0.6B"
    generate_model: str = "Gemma4 26A4B"

    # Vector Storage Configurations
    vector_quantization: str = "none"  # Options: "none" (float32), "int8", "bit" (or "binary")

    # Chunking & Parsing Configurations
    target_chunk_size: int = 1024
    max_chunk_size: int = 2048
    strip_links: bool = True

    # spaCy FTS Settings
    spacy_model: str = "en_core_web_sm"

    # Search Limits
    fts_limit: int = 50
    vec_limit: int = 50
    rerank_candidates: int = 20
    default_limit: int = 10

    @classmethod
    def from_dict(cls, data: Dict, config_path: Optional[Path] = None) -> 'Config':
        collections_raw = data.get('collections', {})
        collections = {}
        for name, cfg in collections_raw.items():
            exts = cfg.get('file_extensions') or cfg.get('formats')
            if isinstance(exts, str):
                exts = [exts]
            collections[name] = CollectionConfig(
                path=cfg.get('path', ''),
                glob=cfg.get('glob', '**/*.md'),
                contexts=cfg.get('contexts'),
                file_extensions=exts,
                convert_non_md=cfg.get('convert_non_md', True)
            )
        
        # Priority for db_path: Environment Var > YAML > Default
        db_path_raw = os.environ.get("QMD_DB_PATH") or data.get('db_path')
        if db_path_raw:
            p = Path(db_path_raw).expanduser()
            if not p.is_absolute() and config_path:
                db_path = str((config_path.parent / p).resolve())
            else:
                db_path = str(p.resolve())
        elif config_path and config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            db_path = str((config_path.parent / "qmd.db").resolve())
        else:
            db_path = str((Path.home() / ".config" / "qmd" / "qmd.db").resolve())

        # Priority for history_db_path: Environment Var > YAML > Default
        history_db_path_raw = os.environ.get("QMD_HISTORY_DB_PATH") or data.get('history_db_path')
        if history_db_path_raw:
            p = Path(history_db_path_raw).expanduser()
            if not p.is_absolute() and config_path:
                history_db_path = str((config_path.parent / p).resolve())
            else:
                history_db_path = str(p.resolve())
        elif config_path and config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            history_db_path = str((config_path.parent / "qmd-history.db").resolve())
        else:
            history_db_path = str((Path.home() / ".config" / "qmd" / "qmd-history.db").resolve())

        # Priority: Environment Var > YAML > Default
        llm_url = os.environ.get("QMD_LLM_URL") or data.get("llm_url") or "http://127.0.0.1:8888"
        api_key = os.environ.get("QMD_LLM_API_KEY") or data.get("api_key")

        embed_url = os.environ.get("QMD_EMBED_URL") or data.get("embed_url")
        rerank_url = os.environ.get("QMD_RERANK_URL") or data.get("rerank_url")
        embed_api_key = os.environ.get("QMD_EMBED_API_KEY") or data.get("embed_api_key")
        rerank_api_key = os.environ.get("QMD_RERANK_API_KEY") or data.get("rerank_api_key")
        
        req_timeout_env = os.environ.get("QMD_REQUEST_TIMEOUT")
        request_timeout = float(req_timeout_env) if req_timeout_env else float(data.get("request_timeout", 120.0))
        
        batch_size_env = os.environ.get("QMD_EMBED_BATCH_SIZE")
        embed_batch_size = int(batch_size_env) if batch_size_env else int(data.get("embed_batch_size", 16))

        embed_model = os.environ.get("EMBED_MODEL") or data.get("embed_model") or "EmbeddingGemma 300m"
        rerank_model = os.environ.get("RERANK_MODEL") or data.get("rerank_model") or "Qwen Rerank 0.6B"
        generate_model = os.environ.get("GENERATE_MODEL") or data.get("generate_model") or "Gemma4 26A4B"
        vector_quantization = os.environ.get("QMD_VECTOR_QUANTIZATION") or data.get("vector_quantization") or "none"
        spacy_model = os.environ.get("SPACY_MODEL") or data.get("spacy_model") or "en_core_web_sm"

        return cls(
            collections=collections,
            db_path=db_path,
            history_db_path=history_db_path,
            config_path=str(config_path.resolve()) if config_path else None,
            llm_url=llm_url,
            api_key=api_key,
            embed_url=embed_url,
            rerank_url=rerank_url,
            embed_api_key=embed_api_key,
            rerank_api_key=rerank_api_key,
            request_timeout=request_timeout,
            embed_batch_size=embed_batch_size,
            embed_model=embed_model,
            rerank_model=rerank_model,
            generate_model=generate_model,
            vector_quantization=vector_quantization,
            target_chunk_size=data.get('target_chunk_size', 1024),
            max_chunk_size=data.get('max_chunk_size', 2048),
            strip_links=data.get('strip_links', True),
            spacy_model=spacy_model,
            fts_limit=data.get('fts_limit', 50),
            vec_limit=data.get('vec_limit', 50),
            rerank_candidates=data.get('rerank_candidates', 20),
            default_limit=data.get('default_limit', 10)
        )

def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """
    Reads configuration from YAML file. Returns default empty config if not found.
    Path resolution order: passed path -> QMD_CONFIG / QMD_CONFIG_PATH env var -> DEFAULT_CONFIG_PATH
    """
    if path is None:
        env_path = os.environ.get("QMD_CONFIG") or os.environ.get("QMD_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path).expanduser().resolve()
        else:
            config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        return Config.from_dict({}, config_path=config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return Config.from_dict(data, config_path=config_path)
    except Exception as e:
        print(f"Warning: Failed to load config at {config_path}: {e}")
        return Config.from_dict({}, config_path=config_path)
