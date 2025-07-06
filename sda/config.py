# sda/config.py

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from sda.core.config_models import AIConfigModel, LLMConfig, LLMRateLimit, EmbeddingConfig, AIProviderConfig

# Load environment variables from .env file
load_dotenv()

# --- Global Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s'
)

# --- Core Configuration ---
WIPE_SCHEMAS_ON_START = os.getenv("WIPE_SCHEMAS_ON_START", "True").lower() in ('true', '1', 't', True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Core Paths ---
DATA_DIR = Path("./data")
WORKSPACE_DIR = str(DATA_DIR / "workspace")

# --- Agent Configuration ---
MAX_TOOL_OUTPUT_LENGTH = int(os.getenv("SDA_MAX_TOOL_OUTPUT_LENGTH", "16000"))
BACKUP_DIR = str(DATA_DIR / "backups")
INGESTION_CACHE_DIR = str(DATA_DIR / "ingestion_cache")

# --- PostgreSQL Database Configuration ---
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB_NAME = os.getenv("PG_DB_NAME", "ami_sda_db")
DB_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB_NAME}"
MAINTENANCE_DB_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/postgres"

# --- Dgraph Configuration ---
DGRAPH_HOST = os.getenv("DGRAPH_HOST", "127.0.0.1")
DGRAPH_PORT = os.getenv("DGRAPH_PORT", "9080")

# --- AI Model Configuration (using Pydantic models) ---
# Central, type-safe configuration for all AI models.
# This single object is the source of truth.
MASTER_AI_CONFIG = AIConfigModel(
    google=AIProviderConfig(
        llms={
            "gemini-2.5-flash-lite-preview-06-17": LLMConfig(
                model_name="gemini-2.5-flash-lite-preview-06-17",
                max_context_tokens=131072,
                soft_context_limit_tokens=16384,  # ~12.5% of max
                rate_limits=[LLMRateLimit(requests=1, period_seconds=1), LLMRateLimit(requests=60, period_seconds=60)],
                input_price_per_million_tokens=0.35,
                output_price_per_million_tokens=1.05,
                provider="google"
            ),
            "gemma-3-27b-it": LLMConfig(
                model_name="gemma-3-27b-it",
                max_context_tokens=131072,
                soft_context_limit_tokens=16384,
                rate_limits=[LLMRateLimit(requests=0.5, period_seconds=1),
                             LLMRateLimit(requests=30, period_seconds=60)],
                input_price_per_million_tokens=0.0,
                output_price_per_million_tokens=0.0,
                provider="google"
            ),
        }
    ),
    local=AIProviderConfig(
        embeddings={
            "alikia2x/jina-embedding-v3-m2v-1024": EmbeddingConfig(
                model_name="alikia2x/jina-embedding-v3-m2v-1024",
                max_tokens=1024,
                dimension=1024,
                # Placeholder cost for local model based on estimated hardware/power usage.
                # E.g., ~$0.02 per million tokens on a mid-range GPU.
                price_per_million_tokens=0.02,
                provider="local"
            )
        }
    )
)


class AIConfig:
    """A static class to provide convenient access to the active model configurations."""
    # Define the active models to be used by the application
    ACTIVE_LLM_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    ACTIVE_EMBEDDING_MODEL = "alikia2x/jina-embedding-v3-m2v-1024"

    # --- LLM Access ---
    @classmethod
    def get_active_llm_config(cls) -> LLMConfig:
        """Returns the full configuration object for the currently active LLM."""
        for provider_conf in MASTER_AI_CONFIG.model_dump().values():
            if cls.ACTIVE_LLM_MODEL in provider_conf.get('llms', {}):
                return LLMConfig(**provider_conf['llms'][cls.ACTIVE_LLM_MODEL])
        raise ValueError(f"Active LLM '{cls.ACTIVE_LLM_MODEL}' not found in any provider configuration.")

    @classmethod
    def get_all_llm_configs(cls) -> dict[str, LLMConfig]:
        """Returns a unified dictionary of all configured LLMs."""
        all_llms = {}
        for provider_conf in MASTER_AI_CONFIG.model_dump().values():
            all_llms.update(provider_conf.get('llms', {}))
        return {name: LLMConfig(**conf) for name, conf in all_llms.items()}

    # --- Embedding Access ---
    @classmethod
    def get_active_embedding_config(cls) -> EmbeddingConfig:
        """Returns the full configuration object for the currently active embedding model."""
        for provider_conf in MASTER_AI_CONFIG.model_dump().values():
            if cls.ACTIVE_EMBEDDING_MODEL in provider_conf.get('embeddings', {}):
                return EmbeddingConfig(**provider_conf['embeddings'][cls.ACTIVE_EMBEDDING_MODEL])
        raise ValueError(
            f"Active embedding model '{cls.ACTIVE_EMBEDDING_MODEL}' not found in any provider configuration.")

    # --- General Properties ---
    EMBEDDING_DEVICES = ["auto"]
    MAX_EMBEDDING_WORKERS = 2
    VECTOR_COLLECTION_NAME = "code_chunks"


# --- Chunking & Ingestion Configuration ---
class IngestionConfig:
    """Configuration for the ingestion and chunking process."""
    # Maps file extensions to their tree-sitter language name.
    LANGUAGE_MAPPING = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java',
        '.sh': 'bash', '.bash': 'bash',
        '.md': 'markdown', '.markdown': 'markdown',
        '.html': 'html', '.htm': 'html',
        '.css': 'css',
    }
    LANGUAGE_SETTINGS = {
        'python': {
            'CHUNKABLE_NODES': {'function_definition', 'class_definition'},
            'IDENTIFIER_NODES': {'function_definition', 'class_definition', 'call', 'import_statement',
                                 'import_from_statement'},
            'COMPLEXITY_NODES': {'if_statement', 'for_statement', 'while_statement', 'with_statement', 'except_clause',
                                 'assert_statement', 'raise_statement', 'boolean_operator', 'binary_operator', 'await'}
        },
        'java': {
            'CHUNKABLE_NODES': {'class_declaration', 'method_declaration', 'constructor_declaration',
                                'interface_declaration', 'enum_declaration'},
            'IDENTIFIER_NODES': {'class_declaration', 'method_declaration', 'constructor_declaration',
                                 'method_invocation', 'import_declaration'},
            'COMPLEXITY_NODES': {'if_statement', 'for_statement', 'while_statement', 'do_statement',
                                 'switch_expression', 'catch_clause', 'throw_statement', 'assert_statement',
                                 'synchronized_statement', 'conditional_expression'}
        },
        'javascript': {
            'CHUNKABLE_NODES': {'function_declaration', 'class_declaration', 'method_definition',
                                'lexical_declaration'},
            'IDENTIFIER_NODES': {'function_declaration', 'class_declaration', 'call_expression', 'import_statement'},
            'COMPLEXITY_NODES': {'if_statement', 'for_statement', 'for_in_statement', 'while_statement', 'do_statement',
                                 'switch_statement', 'catch_clause', 'throw_statement', 'await_expression',
                                 'ternary_expression'}
        },
        'typescript': {
            'CHUNKABLE_NODES': {'function_declaration', 'class_declaration', 'method_definition', 'lexical_declaration',
                                'interface_declaration', 'type_alias_declaration'},
            'IDENTIFIER_NODES': {'function_declaration', 'class_declaration', 'call_expression', 'import_statement',
                                 'interface_declaration'},
            'COMPLEXITY_NODES': {'if_statement', 'for_statement', 'for_in_statement', 'while_statement', 'do_statement',
                                 'switch_statement', 'catch_clause', 'throw_statement', 'await_expression',
                                 'ternary_expression'}
        },
        'bash': {
            'CHUNKABLE_NODES': {'function_definition', 'command_substitution'}, # `command` itself is too broad.
            'IDENTIFIER_NODES': {'function_definition', 'variable_assignment', 'command_name'},
            'COMPLEXITY_NODES': {'if_statement', 'for_statement', 'while_statement', 'case_statement', 'pipeline'}
        },
        'markdown': {
            # Markdown chunking might be better handled by semantic splitting or custom logic.
            # These are placeholders if tree-sitter based chunking is attempted.
            'CHUNKABLE_NODES': {'section', 'paragraph', 'block_quote', 'list_item', 'fenced_code_block'},
            'IDENTIFIER_NODES': {'link_destination', 'image_description', 'atx_heading', 'setext_heading'},
            'COMPLEXITY_NODES': {'fenced_code_block'} # e.g. if it contains complex code
        },
        'html': {
            'CHUNKABLE_NODES': {'element', 'script_element', 'style_element'},
            'IDENTIFIER_NODES': {'attribute_value', 'tag_name', 'script_element', 'style_element'}, # id/class often in attribute_value
            'COMPLEXITY_NODES': {'script_element', 'style_element'}
        },
        'css': {
            'CHUNKABLE_NODES': {'rule_set', 'style_rule', 'media_statement'},
            'IDENTIFIER_NODES': {'class_selector', 'id_selector', 'custom_property_name', 'tag_name', 'feature_name'},
            'COMPLEXITY_NODES': {'calc_function', 'media_query', 'supports_condition'}
        }
    }
    DEFAULT_IGNORE_DIRS = {".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode",
                           "target", "docs"}
    DEFAULT_IGNORE_FILES = {".DS_Store"}
    SUPPORTED_EXTENSIONS = {
        ".py", ".md", ".txt", ".js", ".ts", ".java",  # Existing
        ".sh", ".bash", ".markdown", ".html", ".htm", ".css"  # New
    }
    TOKENIZER_MODEL = "cl100k_base"
    MAX_CHUNK_TOKENS = AIConfig.get_active_embedding_config().max_tokens
    CHUNK_OVERLAP_RATIO = 0.1
    EMBEDDING_TARGET_BATCH_TOKENS = 65536 * 8
    MAX_CPU_WORKERS = 48
    # Configurable per-target worker pools for I/O
    MAX_DB_WORKERS_PER_TARGET = {
        "postgres": int(os.getenv("MAX_POSTGRES_WORKERS", "48")),
        "dgraph": int(os.getenv("MAX_DGRAPH_WORKERS", "48")),
    }
    POOL_SIZE = sum(MAX_DB_WORKERS_PER_TARGET.values())  # Total pool size for engine config
    POOL_TIMEOUT = 60
    FILE_PROCESSING_BATCH_SIZE = 128
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "65536"))
    VECTOR_UPDATE_BATCH_SIZE = int(os.getenv("VECTOR_UPDATE_BATCH_SIZE", "65536"))
    DGRAPH_BATCH_SIZE = int(os.getenv("DGRAPH_BATCH_SIZE", "512"))
    DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "2048"))
    IMPORTANCE_WEIGHTS = {
        "base_score": 0.5, "score_split_structural": 0.4, "score_remainder": 0.2,
        "type_class": 0.2, "type_function": 0.1, "has_docstring": 0.3,
        "high_complexity": 0.15, "is_short": -0.1, "is_long": 0.1,
        "complexity_threshold": 5, "short_chunk_threshold": 20, "long_chunk_threshold": 200,
    }


# --- Smart Partitioning Configuration ---
class PartitioningConfig:
    """Configuration for the smart schema partitioning algorithm."""
    # The ideal number of files for a single schema partition.
    TARGET_FILES_PER_SCHEMA = int(os.getenv("TARGET_FILES_PER_SCHEMA", "500"))
    # The absolute minimum number of files a directory must have to be considered a partition.
    MIN_SCHEMA_FILE_COUNT_ABSOLUTE = int(os.getenv("MIN_SCHEMA_FILE_COUNT_ABSOLUTE", "20"))
    # A ratio of total repository files used to calculate a dynamic minimum.
    # The actual minimum will be max(ABSOLUTE, total_files * RATIO).
    MIN_SCHEMA_FILE_COUNT_RATIO = float(os.getenv("MIN_SCHEMA_FILE_COUNT_RATIO", "0.002"))  # 0.2% of total files
    # The partition search depth is determined by average_depth * MULTIPLIER, capped by an absolute max.
    MAX_DEPTH_AVG_MULTIPLIER = float(os.getenv("MAX_DEPTH_AVG_MULTIPLIER", "1.5"))
    # An absolute cap on the search depth to prevent excessive partitioning in unusually deep repositories.
    MAX_DEPTH_ABSOLUTE_CAP = int(os.getenv("MAX_DEPTH_ABSOLUTE_CAP", "7"))


# --- UI Configuration ---
class UIConfig:
    """Configuration for the Gradio User Interface."""
    VSCODE_SERVER_URL = os.getenv("VSCODE_SERVER_URL", "http://localhost:8080")
