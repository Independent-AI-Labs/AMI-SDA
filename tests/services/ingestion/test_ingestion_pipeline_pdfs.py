# tests/services/ingestion/test_ingestion_pipeline_pdfs.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import asyncio # Required for async functions and asyncio.run
import concurrent.futures # For as_completed used in pipeline

from sda.services.ingestion.pipeline import IntelligentIngestionService
# Import PDFParsingService to mock its methods if needed by worker, or to pass its config
from sda.services.pdf_parser import PDFParsingService
from sda.core.db_management import DatabaseManager
from sda.services.git_integration import GitService
from sda.utils.task_executor import TaskExecutor # TaskExecutor is used
from sda.services.partitioning import SmartPartitioningService
from sda.config import IngestionConfig, AIConfig # For default ignore dirs and other configs

# Dummy data for repository identification
MOCK_REPO_ID = 1
MOCK_REPO_UUID = "repo-uuid-for-pdf-ingestion-test"
MOCK_BRANCH_NAME = "pdf-ingestion-branch"

# This is the actual worker function we want to patch to observe its calls
# It lives in sda.services.ingestion.workers
WORKER_PATH_TO_PATCH = 'sda.services.ingestion.pipeline._process_single_pdf_worker'


@pytest.fixture
def mock_services_for_ingestion_pipeline_test(tmp_path):
    # Mock DatabaseManager
    db_manager = MagicMock(spec=DatabaseManager)
    mock_session = MagicMock()
    db_manager.get_session.return_value.__enter__.return_value = mock_session
    db_manager.db_url = "mock_db_url_for_worker" # For worker to use
    # Mock methods called during ingestion setup/cleanup phase by pipeline
    mock_session.query.return_value.filter.return_value.delete.return_value = None
    mock_session.commit.return_value = None
    db_manager._wipe_repo_schemas.return_value = None
    db_manager.clear_dgraph_data_for_branch.return_value = None
    db_manager.create_schema_and_tables = MagicMock() # Called for code partitions

    # Mock GitService
    git_service = MagicMock(spec=GitService)
    git_service.checkout.return_value = True

    # Mock TaskExecutor: make submit execute the target function immediately (synchronously for test simplicity)
    task_executor = MagicMock(spec=TaskExecutor)
    def immediate_sync_submit_effect(pool_name, target_fn, *args, **kwargs):
        # print(f"Mock TaskExecutor: Submitting {target_fn.__name__} with args {args}")
        future = concurrent.futures.Future()
        try:
            # If target_fn is async, we need to run it in an event loop
            if asyncio.iscoroutinefunction(target_fn):
                # This is tricky in a sync test function if the target is truly async
                # For _process_single_pdf_worker, it internally uses asyncio.run()
                # So, calling it directly as a sync function should work if the worker is designed that way.
                # Let's assume the worker handles its own async context if called synchronously.
                # The worker itself calls asyncio.run(pdf_parser_instance.parse_pdf(...))
                result = target_fn(*args, **kwargs)
            else:
                result = target_fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future
    task_executor.submit = MagicMock(side_effect=immediate_sync_submit_effect)

    # Mock SmartPartitioningService
    partitioning_service = MagicMock(spec=SmartPartitioningService)
    partitioning_service.generate_schema_map.return_value = {} # No code partitions for this test

    # Mock PDFParsingService (the instance that will be on IngestionService)
    pdf_parsing_service_on_ingestion_svc = MagicMock(spec=PDFParsingService)
    pdf_parsing_service_on_ingestion_svc.mineru_path = "mocked_mineru_path_for_worker"

    # Create IntelligentIngestionService instance with mocked dependencies
    ingestion_service = IntelligentIngestionService(
        db_manager=db_manager,
        git_service=git_service,
        task_executor=task_executor,
        partitioning_service=partitioning_service,
        pdf_parsing_service=pdf_parsing_service_on_ingestion_svc # This is passed to worker
    )

    # Also need to mock AIConfig for embedding settings if _setup_embedding_workers is called
    # and not fully patched out.
    with patch('sda.services.ingestion.pipeline.AIConfig', MagicMock(spec=AIConfig)) as MockAIConfig:
        MockAIConfig.get_active_embedding_config.return_value = MagicMock(model_name="test_embed_model", provider="local", price_per_million_tokens=0)
        MockAIConfig.MAX_EMBEDDING_WORKERS = 1
        # Patch resolve_embedding_devices to prevent actual hardware checks
        with patch('sda.services.ingestion.pipeline.resolve_embedding_devices', return_value=["cpu"]):
             yield ingestion_service, db_manager, pdf_parsing_service_on_ingestion_svc, git_service


def create_dummy_repo_files(base_dir: Path, files_to_create: Dict[str, str]):
    for rel_path, content in files_to_create.items():
        full_path = base_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content if content is not None else "dummy content")

# This is the function that will be patched for _process_single_pdf_worker
# It needs to be awaitable if the calling code awaits its future.
# However, our immediate_sync_submit_effect makes the future resolve synchronously.
# The original worker _process_single_pdf_worker itself uses asyncio.run.
# So, the mock can be a simple synchronous function.
def mock_pdf_worker_target_function(*args, **kwargs):
    # Simulate a successful PDF processing
    # args are: absolute_pdf_path, repo_id, branch_name, relative_path, db_url, mineru_path
    # print(f"Mock _process_single_pdf_worker (target) called with: {args}")
    return {"status": "success", "file": args[3], "uuid": f"mock-uuid-for-{args[3]}", "path": args[0]}


@patch(WORKER_PATH_TO_PATCH, new=mock_pdf_worker_target_function) # Patch the actual worker function
def test_ingestion_pipeline_processes_pdf_files(
    mock_services_for_ingestion_pipeline_test,
    tmp_path
):
    ingestion_service, db_manager_mock, _, _ = mock_services_for_ingestion_pipeline_test

    repo_test_path = tmp_path / "pdf_ingestion_repo"
    repo_test_path.mkdir()

    files_in_repo_structure = {
        "document1.pdf": "PDF one content",
        "src/code.py": "print('Python code')",
        "archive/old_document.pdf": "PDF two content",
        "README.md": "Project readme" # This is a code file by default config
    }
    create_dummy_repo_files(repo_test_path, files_in_repo_structure)

    # Mock framework's task update methods (these are passed as callbacks)
    mock_fw_start_task = MagicMock(return_value=MagicMock(id=1001))
    mock_fw_update_task = MagicMock()
    mock_fw_complete_task = MagicMock()

    # Mock the embedding worker setup/shutdown as they involve multiprocessing
    # that can be complex to manage in unit tests if not the focus.
    with patch.object(ingestion_service, '_setup_embedding_workers', return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
         patch.object(ingestion_service, '_shutdown_embedding_workers'):

        ingestion_service.ingest_repository(
            repo_path_str=str(repo_test_path),
            repo_uuid=MOCK_REPO_UUID,
            branch=MOCK_BRANCH_NAME,
            repo_id=MOCK_REPO_ID,
            parent_task_id=99, # Dummy parent task ID
            _framework_start_task=mock_fw_start_task,
            _framework_update_task=mock_fw_update_task,
            _framework_complete_task=mock_fw_complete_task
        )

    # Assertions: Check if task_executor.submit was called for PDF files with _process_single_pdf_worker
    submit_calls = ingestion_service.task_executor.submit.call_args_list

    pdf_worker_submit_calls = [
        call for call in submit_calls
        if call[0][1] == mock_pdf_worker_target_function # Check if the target was our patched worker
    ]
    assert len(pdf_worker_submit_calls) == 2 # Expecting two PDFs

    # Check arguments for each PDF call
    expected_pdf_calls_args = {
        "document1.pdf": (
            str((repo_test_path / "document1.pdf").resolve()),
            MOCK_REPO_ID, MOCK_BRANCH_NAME, "document1.pdf",
            db_manager_mock.db_url,
            ingestion_service.pdf_parsing_service.mineru_path
        ),
        "archive/old_document.pdf": (
            str((repo_test_path / "archive/old_document.pdf").resolve()),
            MOCK_REPO_ID, MOCK_BRANCH_NAME, "archive/old_document.pdf",
            db_manager_mock.db_url,
            ingestion_service.pdf_parsing_service.mineru_path
        ),
    }

    for call in pdf_worker_submit_calls:
        # args passed to submit: (pool_name, target_fn, actual_arg1, actual_arg2, ...)
        passed_args_to_worker = call[0][2:] # These are the args for _process_single_pdf_worker
        relative_path_in_call = passed_args_to_worker[3] # relative_path is the 4th arg to worker (index 3)

        assert relative_path_in_call in expected_pdf_calls_args
        assert passed_args_to_worker == expected_pdf_calls_args[relative_path_in_call]
        del expected_pdf_calls_args[relative_path_in_call] # Mark as found

    assert not expected_pdf_calls_args, f"Some expected PDF calls were not made: {expected_pdf_calls_args.keys()}"

    # Check that main task was completed
    mock_fw_complete_task.assert_called_with(99, result=pytest.ANY) # Check parent task ID
```
