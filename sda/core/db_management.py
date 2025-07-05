# sda/core/db_management.py

"""
Provides a manager class for handling database sessions and graph persistence.

This module contains the DatabaseManager, which is responsible for:
1. Initializing the database engine and creating tables based on the ORM models.
2. Providing a context-managed session for safe database transactions.
3. Managing connections and operations with a Dgraph instance for graph data.
"""

import json
import logging
import random
import time
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, List

import pydgraph
from sqlalchemy import create_engine, text, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from sda.config import (
    DB_URL,
    MAINTENANCE_DB_URL,
    PG_DB_NAME,
    WIPE_SCHEMAS_ON_START,
    DGRAPH_HOST,
    DGRAPH_PORT,
    IngestionConfig,
)
from sda.core.models import Base

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _register_vector_on_connect(dbapi_connection, connection_record):
    """Event listener to register the pgvector type on each new connection."""
    from pgvector.psycopg2 import register_vector
    # The raw DBAPI connection object is dbapi_connection.
    register_vector(dbapi_connection)
    logging.debug("pgvector type registered for new connection.")


class DatabaseManager:
    """Manages connections and schemas for both PostgreSQL and Dgraph."""

    def __init__(self, db_url: str = DB_URL, is_worker: bool = False):
        self.db_url = db_url
        self.is_worker = is_worker
        self.dgraph_client: Optional[pydgraph.DgraphClient] = None

        # The main process handles initialization and schema wiping.
        if not self.is_worker:
            self._ensure_database_and_extension_exist()
            self._connect_dgraph()  # Connects and sets up schema if needed.

        # Workers use a NullPool to avoid sharing connections across processes.
        # The main process uses a QueuePool for efficient connection handling.
        engine_args = {
            "pool_pre_ping": True,
            "connect_args": {'options': '-csearch_path=public'},
            "poolclass": NullPool if self.is_worker else QueuePool,
        }
        if not self.is_worker:
            engine_args.update({
                "pool_size": IngestionConfig.POOL_SIZE,
                "pool_timeout": IngestionConfig.POOL_TIMEOUT
            })

        self.engine = create_engine(self.db_url, **engine_args)

        # The vector type must be registered on each new connection.
        event.listen(self.engine, "connect", _register_vector_on_connect)

        if not self.is_worker:
            if WIPE_SCHEMAS_ON_START:
                self._wipe_all_repo_schemas()
            self._initialize_storage()

    def _ensure_database_and_extension_exist(self):
        """Creates the main database and vector extension if they don't exist."""
        try:
            # First, connect to the maintenance DB to create the main application DB.
            maint_engine = create_engine(MAINTENANCE_DB_URL, isolation_level="AUTOCOMMIT", poolclass=NullPool)
            with maint_engine.connect() as conn:
                exists = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{PG_DB_NAME}'")).scalar()
                if not exists:
                    logging.info(f"Database '{PG_DB_NAME}' not found. Creating it...")
                    conn.execute(text(f'CREATE DATABASE "{PG_DB_NAME}"'))
                    logging.info(f"Database '{PG_DB_NAME}' created successfully.")
            maint_engine.dispose()

            # Second, connect to the app DB to create the extension BEFORE the main engine
            # with the event listener is initialized. This prevents the listener from
            # failing on first connect.
            app_db_engine = create_engine(self.db_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
            with app_db_engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS public;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;"))
            app_db_engine.dispose()
            logging.info(f"Ensured 'public' schema and 'vector' extension exist in database '{PG_DB_NAME}'.")
        except Exception as e:
            logging.error(f"Critical database or extension setup failed: {e}", exc_info=True)
            raise

    def _connect_dgraph(self):
        """Connects to Dgraph and sets up the schema for the main process."""
        try:
            stub_opts = [
                ('grpc.max_send_message_length', 104857600),
                ('grpc.max_receive_message_length', 104857600)
            ]
            client_stub = pydgraph.DgraphClientStub(f"{DGRAPH_HOST}:{DGRAPH_PORT}", options=stub_opts)
            self.dgraph_client = pydgraph.DgraphClient(client_stub)
            logging.info(f"Connected to Dgraph at {DGRAPH_HOST}:{DGRAPH_PORT}")
            self._setup_dgraph_schema()
        except Exception as e:
            logging.error(f"Dgraph connection failed: {e}. Graph features will be unavailable.", exc_info=True)
            self.dgraph_client = None

    def _setup_dgraph_schema(self):
        """Defines and applies the Dgraph schema for AST nodes and their relationships."""
        if not self.dgraph_client: return
        try:
            schema = """
            node_id: string @index(exact) @upsert .
            repo_id: string @index(exact) .
            branch: string @index(exact) .
            node_type: string @index(exact) .
            name: string @index(hash) .
            file_path: string .
            calls: [uid] @reverse .

            type ASTNode {
                node_id
                repo_id
                branch
                node_type
                name
                file_path
                calls
            }
            """
            self.dgraph_client.alter(pydgraph.Operation(schema=schema))
            logging.info("Dgraph schema setup completed.")
        except Exception as e:
            logging.warning(f"Failed to setup Dgraph schema: {e}")

    def _wipe_all_repo_schemas(self):
        """Drops all 'repo_*' schemas and wipes Dgraph. For development/testing only."""
        logging.warning("WIPE_SCHEMAS_ON_START is True. Wiping all repository data.")
        # Wipe Dgraph first
        if self.dgraph_client:
            try:
                self.dgraph_client.alter(pydgraph.Operation(drop_all=True))
                self._setup_dgraph_schema()
                logging.info("Wiped all data from Dgraph.")
            except Exception as e:
                logging.error(f"Failed to wipe Dgraph: {e}")

        # Wipe Postgres schemas and public tables
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            repo_schemas_result = conn.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'repo_%'"))
            for schema in [row[0] for row in repo_schemas_result]:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                logging.info(f"Dropped schema: {schema}")

            logging.info("Dropping and recreating public schema...")
            conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;"))
            logging.info("Public schema recreated.")

    def _initialize_storage(self):
        """Creates the public schema and base tables if they don't exist."""
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
            # This is redundant if _ensure_database_and_extension_exist ran, but harmless
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"))
        Base.metadata.create_all(self.engine)
        logging.info("Initialized public schema and storage, including BillingUsage table.")

    @contextmanager
    def get_session(self, schema_name: str = "public") -> Generator[Session, None, None]:
        """Provides a transactional session scoped to a specific schema."""
        SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=False)
        session = SessionLocal()
        try:
            # Set the search path for this specific transaction only.
            session.execute(text(f"SET LOCAL search_path = '{schema_name}', public"))
            yield session
            session.commit()
        except Exception:
            logging.error(f"Session error on schema '{schema_name}', rolling back.", exc_info=True)
            session.rollback()
            raise
        finally:
            session.close()

    def _wipe_repo_schemas(self, repo_uuid: str):
        """Drops all schemas associated with a specific repository UUID."""
        schema_pattern = f"repo_{repo_uuid[:8]}_%"
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            result = conn.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE :pattern"), {'pattern': schema_pattern})
            for schema in [row[0] for row in result]:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                logging.info(f"Dropped schema: {schema}")

    def create_schema_and_tables(self, schema_name: str):
        """
        Atomically creates a schema and all tables defined in models.py within it.
        This method is resilient to race conditions in concurrent environments.
        """
        try:
            with self.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                try:
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
                except IntegrityError as e:
                    if "already exists" in str(e).lower():
                        logging.debug(f"Schema '{schema_name}' already exists (race condition handled), continuing.")
                        pass
                    else:
                        raise  # Re-raise other integrity errors.
            with self.engine.connect() as conn:
                with conn.begin():  # Transaction for table creation
                    conn.execute(text(f'SET LOCAL search_path = "{schema_name}", public'))
                    Base.metadata.create_all(bind=conn)
            logging.debug(f"Ensured schema and tables exist for: {schema_name}")
        except Exception as e:
            logging.error(f"Failed to create schema or tables for '{schema_name}': {e}", exc_info=True)
            raise

    def clear_dgraph_data_for_branch(self, repo_id: int, branch: str):
        """Deletes all Dgraph nodes associated with a specific repository and branch."""
        if not self.dgraph_client: return
        query = f'{{ nodes(func: eq(repo_id, "{repo_id}")) @filter(eq(branch, "{branch}")) {{ uid }} }}'
        try:
            txn_ro = self.dgraph_client.txn(read_only=True)
            res = json.loads(txn_ro.query(query).json)
            if uids_to_delete := [node['uid'] for node in res.get('nodes', [])]:
                txn_rw = self.dgraph_client.txn()
                for uid in uids_to_delete:
                    txn_rw.mutate(del_nquads=f'<{uid}> * * .')
                txn_rw.commit()
                logging.info(f"Cleared {len(uids_to_delete)} Dgraph nodes for repo {repo_id}, branch {branch}")
        except Exception as e:
            logging.warning(f"Could not clear Dgraph data for repo {repo_id}, branch {branch}: {e}")

    def execute_dgraph_mutations(self, mutations: List[Dict[str, Any]]):
        """Executes a batch of Dgraph mutations with retries for aborted transactions."""
        if not self.dgraph_client or not mutations: return

        MAX_RETRIES = 5
        for i in range(0, len(mutations), IngestionConfig.DGRAPH_BATCH_SIZE):
            batch = mutations[i:i + IngestionConfig.DGRAPH_BATCH_SIZE]
            for attempt in range(MAX_RETRIES):
                txn = self.dgraph_client.txn()
                try:
                    txn.mutate(set_obj=batch)
                    txn.commit()
                    break  # Success
                except pydgraph.errors.AbortedError:
                    logging.warning(f"Dgraph txn aborted. Retrying... ({attempt + 1}/{MAX_RETRIES})")
                    time.sleep(0.1 * (2 ** attempt) + random.uniform(0, 0.1))
                except Exception as e:
                    logging.error(f"Dgraph mutation failed with non-retriable error: {e}", exc_info=True)
                    break  # Don't retry on other errors
                finally:
                    txn.discard()

    def query_dgraph(self, query: str, variables: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Executes a Dgraph query."""
        if not self.dgraph_client: return None
        try:
            res = self.dgraph_client.txn(read_only=True).query(query, variables=variables)
            return json.loads(res.json)
        except Exception as e:
            logging.error(f"Dgraph query failed: {e}", exc_info=True)
            return None

    def get_database_size(self) -> Optional[int]:
        """
        Retrieves the total size of the current PostgreSQL database.

        Returns:
            The database size in bytes, or None if an error occurs.
        """
        try:
            with self.engine.connect() as conn:
                # No need to set search_path for this query as pg_database_size works on current_database()
                result = conn.execute(text("SELECT pg_database_size(current_database())")).scalar_one_or_none()
                return result
        except Exception as e:
            logging.error(f"Failed to get database size: {e}", exc_info=True)
            return None