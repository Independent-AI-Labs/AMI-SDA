# tests/framework/test_app_pdf_processing.py
import pytest
import asyncio
from pathlib import Path
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

from sda.app import CodeAnalysisFramework
from sda.services.pdf_parser import ParsedPDFDocument as PydanticParsedPDFDocument, PDFImageBlob as PydanticPDFImageBlob
from sda.core.models import PDFDocument as SQLAPDFDocument # SQLAlchemy model

# Dummy PDF content for hashing
DUMMY_PDF_CONTENT = b"This is a dummy PDF content."
DUMMY_PDF_HASH = hashlib.sha256(DUMMY_PDF_CONTENT).hexdigest()
DUMMY_PDF_UUID = "test-pdf-uuid-123"
NEW_PDF_UUID = "new-pdf-uuid-456"

@pytest.fixture
def framework_instance(tmp_path):
    # Create a temporary workspace for GitService if it's initialized by default
    # For these tests, we mainly need to mock db_manager and pdf_parsing_service
    # So, we can patch parts of CodeAnalysisFramework.__init__ or its dependencies
    # if their full initialization is problematic for unit testing.

    # For simplicity, let's assume CodeAnalysisFramework can be instantiated,
    # and we will mock its attributes (db_manager, pdf_parsing_service) directly in tests.
    # A more robust approach might involve a test-specific configuration or dependency injection.

    # Create a dummy workspace dir if GitService needs it
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    # Patch DatabaseManager and PDFParsingService instantiation within CodeAnalysisFramework's scope for these tests
    with patch('sda.app.DatabaseManager') as MockDBManager, \
         patch('sda.app.PDFParsingService') as MockPDFParsingService, \
         patch('sda.app.GitService') as MockGitService, \
         patch('sda.app.TaskExecutor'), \
         patch('sda.app.RateLimiter'), \
         patch('sda.app.TokenAwareChunker'), \
         patch('sda.app.FullTextSearchService'), \
         patch('sda.app.EnhancedAnalysisEngine'), \
         patch('sda.app.SmartPartitioningService'), \
         patch('sda.app.IntelligentIngestionService'), \
         patch('sda.app.AdvancedCodeNavigationTools'), \
         patch('sda.app.SafeFileEditingSystem'), \
         patch('sda.app.AgentManager'):

        mock_db_manager = MockDBManager.return_value
        mock_pdf_parser = MockPDFParsingService.return_value

        # Mock the get_session context manager
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        framework = CodeAnalysisFramework(db_url="sqlite:///:memory:", workspace_dir=str(workspace_dir))
        # Assign the mocked instances after __init__ if they were replaced by real ones
        framework.db_manager = mock_db_manager
        framework.pdf_parsing_service = mock_pdf_parser

        return framework, mock_db_manager, mock_pdf_parser, mock_session


@pytest.mark.asyncio
async def test_get_or_process_pdf_existing_document(framework_instance, tmp_path):
    framework, mock_db_manager, _, mock_session = framework_instance

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(DUMMY_PDF_CONTENT)

    # Mock DB response: document exists
    mock_sql_pdf_doc = SQLAPDFDocument(uuid=DUMMY_PDF_UUID, pdf_file_hash=DUMMY_PDF_HASH, repository_id=None)
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_sql_pdf_doc

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file))

    assert result_uuid == DUMMY_PDF_UUID
    mock_session.query.return_value.filter_by.assert_called_once_with(pdf_file_hash=DUMMY_PDF_HASH)
    framework.pdf_parsing_service.parse_pdf.assert_not_called() # Should not parse if found
    mock_db_manager.save_pdf_document.assert_not_called() # Should not save if found (unless linkage update)

@pytest.mark.asyncio
async def test_get_or_process_pdf_existing_document_with_linkage_update(framework_instance, tmp_path):
    framework, mock_db_manager, _, mock_session = framework_instance

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(DUMMY_PDF_CONTENT)

    mock_sql_pdf_doc = SQLAPDFDocument(
        uuid=DUMMY_PDF_UUID,
        pdf_file_hash=DUMMY_PDF_HASH,
        repository_id=None, # Initially no repo link
        branch_name=None,
        relative_path=None
    )
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_sql_pdf_doc

    repo_id = 1
    branch = "main"
    rel_path = "docs/test.pdf"

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file), repo_id, branch, rel_path)

    assert result_uuid == DUMMY_PDF_UUID
    assert mock_sql_pdf_doc.repository_id == repo_id
    assert mock_sql_pdf_doc.branch_name == branch
    assert mock_sql_pdf_doc.relative_path == rel_path
    assert mock_session.commit.called # Commit should be called due to linkage update

@pytest.mark.asyncio
async def test_get_or_process_pdf_new_document(framework_instance, tmp_path):
    framework, mock_db_manager, mock_pdf_parser, mock_session = framework_instance

    pdf_file = tmp_path / "new_test.pdf"
    new_pdf_content = b"new pdf data"
    new_pdf_hash = hashlib.sha256(new_pdf_content).hexdigest()
    pdf_file.write_bytes(new_pdf_content)

    # Mock DB response: document does not exist
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    # Mock PDFParsingService response
    mock_parsed_doc_pydantic = PydanticParsedPDFDocument(pdf_file_hash=new_pdf_hash, total_pages=1, pages=[])
    mock_image_blobs = []
    mock_pdf_parser.parse_pdf = AsyncMock(return_value=(mock_parsed_doc_pydantic, mock_image_blobs))

    # Mock DatabaseManager save response
    mock_db_manager.save_pdf_document.return_value = NEW_PDF_UUID

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file), repo_id=1, branch_name="dev", relative_path="new.pdf")

    assert result_uuid == NEW_PDF_UUID
    mock_session.query.return_value.filter_by.assert_called_once_with(pdf_file_hash=new_pdf_hash)
    mock_pdf_parser.parse_pdf.assert_called_once_with(str(pdf_file))
    mock_db_manager.save_pdf_document.assert_called_once_with(
        parsed_document_data=mock_parsed_doc_pydantic,
        image_blobs_data=mock_image_blobs,
        repository_id=1,
        branch_name="dev",
        relative_path="new.pdf"
    )

@pytest.mark.asyncio
async def test_get_or_process_pdf_file_not_exist(framework_instance):
    framework, _, _, _ = framework_instance
    result_uuid = await framework.get_or_process_pdf_document("/path/to/nonexistent.pdf")
    assert result_uuid is None

@pytest.mark.asyncio
async def test_get_or_process_pdf_hash_mismatch(framework_instance, tmp_path):
    framework, _, mock_pdf_parser, mock_session = framework_instance

    pdf_file = tmp_path / "mismatch.pdf"
    pdf_content = b"actual content"
    calculated_hash = hashlib.sha256(pdf_content).hexdigest()
    pdf_file.write_bytes(pdf_content)

    mock_session.query.return_value.filter_by.return_value.first.return_value = None # Not in DB

    # Parsing service returns a doc with a *different* hash
    mismatched_hash = "differenthash123"
    mock_parsed_doc_pydantic = PydanticParsedPDFDocument(pdf_file_hash=mismatched_hash, total_pages=1, pages=[])
    mock_pdf_parser.parse_pdf = AsyncMock(return_value=(mock_parsed_doc_pydantic, []))

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file))
    assert result_uuid is None
    mock_pdf_parser.parse_pdf.assert_called_once()
    framework.db_manager.save_pdf_document.assert_not_called()

@pytest.mark.asyncio
async def test_get_or_process_pdf_parsing_exception(framework_instance, tmp_path):
    framework, _, mock_pdf_parser, mock_session = framework_instance

    pdf_file = tmp_path / "error.pdf"
    pdf_file.write_bytes(b"error content")

    mock_session.query.return_value.filter_by.return_value.first.return_value = None # Not in DB
    mock_pdf_parser.parse_pdf = AsyncMock(side_effect=Exception("Parsing failed!"))

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file))
    assert result_uuid is None
    framework.db_manager.save_pdf_document.assert_not_called()

@pytest.mark.asyncio
async def test_get_or_process_pdf_db_save_exception(framework_instance, tmp_path):
    framework, mock_db_manager, mock_pdf_parser, mock_session = framework_instance

    pdf_file = tmp_path / "dberror.pdf"
    pdf_content = b"db error content"
    pdf_hash = hashlib.sha256(pdf_content).hexdigest()
    pdf_file.write_bytes(pdf_content)

    mock_session.query.return_value.filter_by.return_value.first.return_value = None # Not in DB

    mock_parsed_doc_pydantic = PydanticParsedPDFDocument(pdf_file_hash=pdf_hash, total_pages=1, pages=[])
    mock_pdf_parser.parse_pdf = AsyncMock(return_value=(mock_parsed_doc_pydantic, []))

    mock_db_manager.save_pdf_document.side_effect = Exception("DB save failed!")

    result_uuid = await framework.get_or_process_pdf_document(str(pdf_file))
    assert result_uuid is None
    mock_db_manager.save_pdf_document.assert_called_once()

```

This test suite covers the main scenarios for `get_or_process_pdf_document`. The `framework_instance` fixture uses patching to mock dependencies of `CodeAnalysisFramework` that are not directly relevant to this method's logic, allowing for more focused unit testing.

Now, I will create this test file.
