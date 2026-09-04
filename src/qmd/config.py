import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Union, Set

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
    include: List[str] = field(default_factory=list)
    is_federated: bool = False
    included_configs: List['Config'] = field(default_factory=list)
    
    # Core LLM Settings
    llm_url: str = "http://127.0.0.1:8888"
    api_key: Optional[str] = None
    embed_url: Optional[str] = None
    rerank_url: Optional[str] = None
    embed_api_key: Optional[str] = None
    rerank_api_key: Optional[str] = None
    request_timeout: float = 120.0
    embed_batch_size: int = 16
    
    # Vision & Multimodal Settings
    vision_url: Optional[str] = None
    vision_api_key: Optional[str] = None
    multimodal_url: Optional[str] = None
    multimodal_api_key: Optional[str] = None
    multimodal_model: Optional[str] = None
    multimodal_prompt: Optional[str] = None
    max_image_concurrency: int = 4
    
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

    # Search Limits & Caching
    cache_search_results: bool = True
    fts_limit: int = 50
    vec_limit: int = 50
    rerank_candidates: int = 20
    default_limit: int = 10

    # System Integration
    allow_open_file: bool = True

    @classmethod
    def from_dict(cls, data: Dict, config_path: Optional[Path] = None, visited_configs: Optional[Set[Path]] = None) -> 'Config':
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
        
        vision_url = os.environ.get("QMD_VISION_URL") or data.get("vision_url")
        vision_api_key = os.environ.get("QMD_VISION_API_KEY") or data.get("vision_api_key")
        
        multimodal_url = os.environ.get("QMD_MULTIMODAL_URL") or data.get("multimodal_url")
        multimodal_api_key = os.environ.get("QMD_MULTIMODAL_API_KEY") or data.get("multimodal_api_key")
        multimodal_model = os.environ.get("QMD_MULTIMODAL_MODEL") or os.environ.get("MULTIMODAL_MODEL") or data.get("multimodal_model")
        multimodal_prompt = os.environ.get("QMD_MULTIMODAL_PROMPT") or data.get("multimodal_prompt")
        
        max_img_concurrency_raw = (
            os.environ.get("QMD_MAX_IMAGE_CONCURRENCY")
            or os.environ.get("QMD_MAX_SIMULTANEOUS_IMAGES")
            or data.get("max_image_concurrency")
            or data.get("max_simultaneous_images")
            or data.get("vision_max_concurrency")
        )
        max_image_concurrency = int(max_img_concurrency_raw) if max_img_concurrency_raw else 4

        if (multimodal_model or data.get("multimodal_url")) and not multimodal_url:
            multimodal_url = llm_url
        if multimodal_url and not multimodal_api_key:
            multimodal_api_key = api_key
        
        req_timeout_env = os.environ.get("QMD_REQUEST_TIMEOUT")
        request_timeout = float(req_timeout_env) if req_timeout_env else float(data.get("request_timeout", 120.0))
        
        batch_size_env = os.environ.get("QMD_EMBED_BATCH_SIZE")
        embed_batch_size = int(batch_size_env) if batch_size_env else int(data.get("embed_batch_size", 16))

        allow_open_env = os.environ.get("QMD_ALLOW_OPEN_FILE") or os.environ.get("QMD_ALLOW_OPEN")
        if allow_open_env is not None:
            allow_open_file = allow_open_env.strip().lower() in ("1", "true", "yes", "on")
        else:
            raw_allow_open = data.get('allow_open_file', data.get('allow_open', True))
            if isinstance(raw_allow_open, str):
                allow_open_file = raw_allow_open.strip().lower() in ("1", "true", "yes", "on")
            else:
                allow_open_file = bool(raw_allow_open)

        embed_model = os.environ.get("EMBED_MODEL") or data.get("embed_model") or "EmbeddingGemma 300m"
        rerank_model = os.environ.get("RERANK_MODEL") or data.get("rerank_model") or "Qwen Rerank 0.6B"
        generate_model = os.environ.get("GENERATE_MODEL") or data.get("generate_model") or "Gemma4 26A4B"
        vector_quantization = os.environ.get("QMD_VECTOR_QUANTIZATION") or data.get("vector_quantization") or "none"
        spacy_model = os.environ.get("SPACY_MODEL") or data.get("spacy_model") or "en_core_web_sm"

        cache_results_env = os.environ.get("QMD_CACHE_SEARCH_RESULTS")
        if cache_results_env is not None:
            cache_search_results = cache_results_env.strip().lower() in ("1", "true", "yes", "on")
        else:
            cache_search_results = data.get('cache_search_results', True)

        current_resolved_path = config_path.resolve() if config_path else None
        active_visited: Set[Path] = set(visited_configs) if visited_configs else set()
        if current_resolved_path:
            active_visited.add(current_resolved_path)

        include_raw = data.get('include') or data.get('includes') or []
        if isinstance(include_raw, str):
            include_raw = [include_raw]
        elif not isinstance(include_raw, list):
            include_raw = []
        include_list: List[str] = [str(item).strip() for item in include_raw if str(item).strip()]

        included_configs: List['Config'] = []
        for inc_item in include_list:
            inc_p = Path(inc_item).expanduser()
            if not inc_p.is_absolute() and config_path:
                inc_resolved = (config_path.parent / inc_p).resolve()
            else:
                inc_resolved = inc_p.resolve()

            if current_resolved_path and inc_resolved == current_resolved_path:
                raise ValueError(f"Self-referential configuration include detected: '{inc_resolved}'")

            if inc_resolved in active_visited:
                raise ValueError(f"Circular configuration include detected: '{inc_resolved}'")

            if not inc_resolved.exists():
                raise FileNotFoundError(f"Included configuration file not found: '{inc_resolved}'")

            child_cfg = load_config(inc_resolved, visited_configs=active_visited)

            # Validate embedding compatibility
            if child_cfg.embed_model != embed_model:
                raise ValueError(
                    f"Embedding model mismatch in included config '{inc_resolved}': "
                    f"master uses '{embed_model}', child uses '{child_cfg.embed_model}'"
                )
            if child_cfg.vector_quantization != vector_quantization:
                raise ValueError(
                    f"Vector quantization mismatch in included config '{inc_resolved}': "
                    f"master uses '{vector_quantization}', child uses '{child_cfg.vector_quantization}'"
                )

            # Validate database path uniqueness
            if child_cfg.db_path and db_path and child_cfg.db_path == db_path:
                raise ValueError(f"Duplicate database path '{child_cfg.db_path}' in included config '{inc_resolved}'")
            for prev_child in included_configs:
                if prev_child.db_path and child_cfg.db_path and prev_child.db_path == child_cfg.db_path:
                    raise ValueError(f"Duplicate database path '{child_cfg.db_path}' across included configs: '{inc_resolved}'")

            # Validate collection name collisions
            for coll_name in child_cfg.collections:
                if coll_name in collections:
                    raise ValueError(f"Duplicate collection name '{coll_name}' in included config '{inc_resolved}' already exists in master configuration")
                for prev_child in included_configs:
                    if coll_name in prev_child.collections:
                        raise ValueError(f"Duplicate collection name '{coll_name}' in included config '{inc_resolved}' already exists in included config '{prev_child.config_path}'")

            # Master LLM endpoints unconditionally override child configuration endpoints
            child_cfg.llm_url = llm_url
            child_cfg.api_key = api_key
            child_cfg.embed_url = embed_url
            child_cfg.rerank_url = rerank_url
            child_cfg.embed_api_key = embed_api_key
            child_cfg.rerank_api_key = rerank_api_key
            child_cfg.vision_url = vision_url
            child_cfg.vision_api_key = vision_api_key
            child_cfg.multimodal_url = multimodal_url
            child_cfg.multimodal_api_key = multimodal_api_key
            child_cfg.multimodal_model = multimodal_model
            child_cfg.multimodal_prompt = multimodal_prompt
            child_cfg.max_image_concurrency = max_image_concurrency
            child_cfg.request_timeout = request_timeout
            child_cfg.embed_batch_size = embed_batch_size
            child_cfg.rerank_model = rerank_model
            child_cfg.generate_model = generate_model
            child_cfg.allow_open_file = allow_open_file

            included_configs.append(child_cfg)

        # Merge included collections into overall collections view
        all_collections = dict(collections)
        for child_cfg in included_configs:
            all_collections.update(child_cfg.collections)

        return cls(
            collections=all_collections,
            db_path=db_path,
            history_db_path=history_db_path,
            config_path=str(config_path.resolve()) if config_path else None,
            include=include_list,
            is_federated=len(included_configs) > 0,
            included_configs=included_configs,
            llm_url=llm_url,
            api_key=api_key,
            embed_url=embed_url,
            rerank_url=rerank_url,
            embed_api_key=embed_api_key,
            rerank_api_key=rerank_api_key,
            vision_url=vision_url,
            vision_api_key=vision_api_key,
            multimodal_url=multimodal_url,
            multimodal_api_key=multimodal_api_key,
            multimodal_model=multimodal_model,
            multimodal_prompt=multimodal_prompt,
            max_image_concurrency=max_image_concurrency,
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
            cache_search_results=cache_search_results,
            fts_limit=data.get('fts_limit', 50),
            vec_limit=data.get('vec_limit', 50),
            rerank_candidates=data.get('rerank_candidates', 20),
            default_limit=data.get('default_limit', 10),
            allow_open_file=allow_open_file
        )

def load_config(path: Optional[Union[str, Path]] = None, visited_configs: Optional[Set[Path]] = None) -> Config:
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
        return Config.from_dict({}, config_path=config_path, visited_configs=visited_configs)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return Config.from_dict(data, config_path=config_path, visited_configs=visited_configs)
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        print(f"Warning: Failed to load config at {config_path}: {e}")
        return Config.from_dict({}, config_path=config_path, visited_configs=visited_configs)
