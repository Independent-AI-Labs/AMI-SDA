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
from sda.core.models import Base, PDFDocument, PDFImageBlobStore # SQLAlchemy models
from sda.services.pdf_parser import ParsedPDFDocument as PydanticParsedPDFDocument
from sda.services.pdf_parser import PDFImageBlob as PydanticPDFImageBlob
from datetime import datetime # For updated_at timestamp

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
        """Deletes all Dgraph nodes associated with a specific repository and branch in batches."""
        if not self.dgraph_client:
            logging.info("Dgraph client not available, skipping clear_dgraph_data_for_branch.")
            return

        batch_size = 10000  # Number of UIDs to fetch and delete per batch
        total_deleted_count = 0
        logging.info(f"Starting batched Dgraph data clearing for repo {repo_id}, branch {branch}, batch size {batch_size}.")

        while True:
            # Query for a batch of UIDs
            query = f'''{{
                nodes(func: eq(repo_id, "{repo_id}"), first: {batch_size}) @filter(eq(branch, "{branch}")) {{
                    uid
                }}
            }}'''

            uids_to_delete = []
            txn_ro = None # Ensure txn_ro is defined for potential discard in finally
            try:
                txn_ro = self.dgraph_client.txn(read_only=True)
                res_json = txn_ro.query(query).json  # Fetch data as string
                res = json.loads(res_json)      # Parse JSON
                uids_to_delete = [node['uid'] for node in res.get('nodes', [])]
            except Exception as e:
                logging.error(f"Dgraph query failed during batch fetch for deletion (repo {repo_id}, branch {branch}): {e}", exc_info=True)
                return # Stop if query fails
            finally:
                if txn_ro:
                    txn_ro.discard()

            if not uids_to_delete:
                logging.info(f"No more Dgraph nodes found to delete for repo {repo_id}, branch {branch}. Loop terminating.")
                break # No more nodes to delete

            txn_rw = None # Ensure txn_rw is defined for potential discard in finally
            try:
                txn_rw = self.dgraph_client.txn()
                # Dgraph recommends using a single mutation with multiple UID dictionaries for deletion.
                # Or a single Del Nquads string.
                # Let's use the multiple UID dicts approach if `delete_obj` is suitable for `S * *` type deletes,
                # otherwise stick to batched N-quads.
                # The `S * *` pattern is best with N-quads.

                nquads_list = [f'<{uid_val}> * * .' for uid_val in uids_to_delete]
                mutation_body = "\n".join(nquads_list)

                # For very large numbers of UIDs, even the mutation request string could be large.
                # We might need to further batch the mutation itself if batch_size for UIDs is too high.
                # For now, 10000 UIDs should result in a manageable N-quads string.
                # Example: <0x123> * * . is ~15 bytes. 10000 * 15 bytes = 150KB, well within limits.

                txn_rw.mutate(del_nquads=mutation_body)
                txn_rw.commit()

                count_in_batch = len(uids_to_delete)
                total_deleted_count += count_in_batch
                logging.info(f"Deleted batch of {count_in_batch} Dgraph nodes for repo {repo_id}, branch {branch}. Total so far: {total_deleted_count}")
            except pydgraph.errors.AbortedError:
                logging.warning(f"Dgraph transaction aborted during batch delete for repo {repo_id}, branch {branch}. Retrying logic might be needed or smaller batches.")
                # For simplicity in this fix, we'll log and stop. A full retry mechanism is more complex.
                return
            except Exception as e:
                logging.error(f"Dgraph batch delete mutation failed (repo {repo_id}, branch {branch}): {e}", exc_info=True)
                return # Stop if mutation fails
            finally:
                if txn_rw and not txn_rw._finished: # Changed .finished to ._finished
                    txn_rw.discard()

        logging.info(f"Successfully cleared a total of {total_deleted_count} Dgraph nodes for repo {repo_id}, branch {branch}.")

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

    def save_pdf_document(
        self,
        parsed_document_data: PydanticParsedPDFDocument,
        image_blobs_data: List[PydanticPDFImageBlob],
        repository_id: Optional[int] = None, # If PDF is associated with a repo
        branch_name: Optional[str] = None,   # If PDF is associated with a repo branch
        relative_path: Optional[str] = None  # If PDF is from a repo
    ) -> Optional[str]:
        """
        Saves a parsed PDF document and its associated image blobs to the database.
        Uses the public schema for these tables. Returns the UUID of the PDFDocument.
        """
        db_pdf_doc_for_return_uuid: Optional[PDFDocument] = None
        with self.get_session(schema_name="public") as session:
            try:
                # 1. Save or update PDFImageBlobs
                for pydantic_blob in image_blobs_data:
                    existing_blob = session.get(PDFImageBlobStore, pydantic_blob.blob_id)
                    if not existing_blob:
                        db_image_blob = PDFImageBlobStore(
                            blob_id=pydantic_blob.blob_id,
                            content_type=pydantic_blob.content_type,
                            data=pydantic_blob.data,
                            size_bytes=len(pydantic_blob.data)
                        )
                        session.add(db_image_blob)

                # Commit image blobs separately to handle potential race if PDF doc save fails
                # or if we need blob IDs before PDF doc is fully committed.
                # However, for atomicity of the operation, usually commit at the very end.
                # Let's try committing at the end first. If there are issues with FKs or
                # unique constraints being violated by concurrent processes, we might reconsider.

                # 2. Save or update PDFDocument
                db_pdf_doc = session.query(PDFDocument).filter_by(pdf_file_hash=parsed_document_data.pdf_file_hash).first()

                if db_pdf_doc:
                    db_pdf_doc.parsed_data = parsed_document_data.model_dump(mode="json")
                    db_pdf_doc.total_pages = parsed_document_data.total_pages
                    db_pdf_doc.repository_id = repository_id
                    db_pdf_doc.branch_name = branch_name
                    db_pdf_doc.relative_path = relative_path
                    db_pdf_doc.updated_at = datetime.utcnow()
                    logging.info(f"Updating existing PDFDocument with hash {parsed_document_data.pdf_file_hash}")
                    db_pdf_doc_for_return_uuid = db_pdf_doc
                else:
                    new_db_pdf_doc = PDFDocument(
                        pdf_file_hash=parsed_document_data.pdf_file_hash,
                        parsed_data=parsed_document_data.model_dump(mode="json"),
                        total_pages=parsed_document_data.total_pages,
                        repository_id=repository_id,
                        branch_name=branch_name,
                        relative_path=relative_path
                    )
                    session.add(new_db_pdf_doc)
                    logging.info(f"Creating new PDFDocument with hash {parsed_document_data.pdf_file_hash}")
                    # We need to get the UUID after adding but before commit for it to be populated if it's server-generated
                    # or ensure it's generated client-side. Our PDFDocument model has a default_factory for uuid.
                    # To get the UUID, we might need to flush or rely on the instance after commit.
                    # For now, let's assign it to the variable that will be used for return.
                    db_pdf_doc_for_return_uuid = new_db_pdf_doc

                session.commit() # Commit all changes (images and document)

                # If db_pdf_doc_for_return_uuid was a new object, its UUID should be populated after commit.
                # If it was an existing one, its UUID is already there.
                if db_pdf_doc_for_return_uuid:
                    logging.info(f"Successfully saved/updated PDF document {parsed_document_data.pdf_file_hash} (UUID: {db_pdf_doc_for_return_uuid.uuid}) and {len(image_blobs_data)} image blobs.")
                    return db_pdf_doc_for_return_uuid.uuid
                return None # Should not happen if commit was successful

            except IntegrityError as e:
                session.rollback()
                logging.error(f"Integrity error while saving PDF document or blobs: {e}", exc_info=True)
                if "pdf_documents_pdf_file_hash_key" in str(e).lower():
                     raise ValueError(f"A PDF document with hash {parsed_document_data.pdf_file_hash} might already exist due to a race condition or was inserted by another process.")
                if "pdf_image_blobs_pkey" in str(e).lower():
                    raise ValueError(f"An image blob with one of the provided blob_ids might already exist due to a race condition.")
                raise
            except Exception as e:
                session.rollback()
                logging.error(f"Error saving PDF document or blobs: {e}", exc_info=True)
                raise
        return None # Should not be reached if logic is correct

    def get_pdf_document_by_hash(self, pdf_file_hash: str) -> Optional[PydanticParsedPDFDocument]:
        """Retrieves a parsed PDF document by its file hash."""
        with self.get_session(schema_name="public") as session:
            db_pdf_doc = session.query(PDFDocument).filter_by(pdf_file_hash=pdf_file_hash).first()
            if db_pdf_doc:
                return PydanticParsedPDFDocument(**db_pdf_doc.parsed_data)
            return None

    def get_pdf_document_by_uuid(self, doc_uuid: str) -> Optional[PydanticParsedPDFDocument]:
        """Retrieves a parsed PDF document by its UUID."""
        with self.get_session(schema_name="public") as session:
            db_pdf_doc = session.query(PDFDocument).filter_by(uuid=doc_uuid).first()
            if db_pdf_doc:
                # Ensure parsed_data is not None and is a dict
                if db_pdf_doc.parsed_data and isinstance(db_pdf_doc.parsed_data, dict):
                    return PydanticParsedPDFDocument(**db_pdf_doc.parsed_data)
                else:
                    logging.error(f"PDFDocument with UUID {doc_uuid} has invalid parsed_data: {db_pdf_doc.parsed_data}")
                    return None
            return None

    def get_pdf_image_blob(self, blob_id: str) -> Optional[PydanticPDFImageBlob]:
        """Retrieves an image blob by its ID."""
        with self.get_session(schema_name="public") as session:
            db_image_blob = session.get(PDFImageBlobStore, blob_id)
            if db_image_blob:
                return PydanticPDFImageBlob(
                    blob_id=db_image_blob.blob_id,
                    content_type=db_image_blob.content_type,
                    data=db_image_blob.data
                )
            return None