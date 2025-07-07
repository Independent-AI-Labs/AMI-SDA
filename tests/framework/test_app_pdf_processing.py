# tests/framework/test_app_pdf_processing.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import hashlib

from sda.app import CodeAnalysisFramework
from sda.core.models import PDFDocument as SQLAPDFDocument, RepositoryPDFLink
from sda.services.pdf_parser import ParsedPDFDocument as PydanticParsedPDFDocument # For type hints if needed elsewhere

DUMMY_PDF_DOC_UUID_FROM_LINK = "retrieved-pdf-doc-uuid-123"

@pytest.fixture
def framework_instance_for_pdf_retrieval(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(exist_ok=True)

    with patch('sda.app.DatabaseManager') as MockDBManager, \
         patch('sda.app.PDFParsingService'), \
         patch('sda.app.GitService'), \
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

        mock_db_manager_instance = MockDBManager.return_value
        mock_session = MagicMock()
        mock_db_manager_instance.get_session.return_value.__enter__.return_value = mock_session

        framework = CodeAnalysisFramework(db_url="sqlite:///:memory:", workspace_dir=str(workspace_dir))
        framework.db_manager = mock_db_manager_instance
        yield framework, mock_session


@pytest.mark.asyncio
async def test_get_pdf_document_uuid_for_repo_file_found(framework_instance_for_pdf_retrieval):
    framework, mock_session = framework_instance_for_pdf_retrieval

    repo_id = 1
    branch = "main"
    rel_path = "docs/report.pdf"

    # Mock the chained query:
    # query(SQLAPDFDocument.uuid).join(RepositoryPDFLink, SQLAPDFDocument.id == RepositoryPDFLink.pdf_document_id)
    # .filter(...).scalar_one_or_none()

    # Mock for scalar_one_or_none()
    mock_scalar_method = MagicMock(return_value=DUMMY_PDF_DOC_UUID_FROM_LINK)

    # Mock for filter()
    mock_filter_obj = MagicMock()
    mock_filter_obj.scalar_one_or_none = mock_scalar_method # filter(...).scalar_one_or_none()

    # Mock for join()
    mock_join_obj = MagicMock()
    mock_join_obj.filter.return_value = mock_filter_obj # join(...).filter(...)

    # Mock for query()
    mock_session.query.return_value.join.return_value = mock_join_obj # query(...).join(...)

    result_uuid = await framework.get_pdf_document_uuid_for_repo_file(repo_id, branch, rel_path)

    assert result_uuid == DUMMY_PDF_DOC_UUID_FROM_LINK
    mock_session.query.assert_called_once_with(SQLAPDFDocument.uuid)

    # Assert that join was called, e.g. with RepositoryPDFLink and the correct join condition
    # This is a bit more involved to assert precisely due to SQLAlchemy's expression objects
    # For now, checking that join was called is a basic step.
    mock_session.query.return_value.join.assert_called_once()

    # Assert that filter was called on the result of join
    mock_join_obj.filter.assert_called_once()
    # We could inspect call_args for filter if needed:
    # filter_args = mock_join_obj.filter.call_args[0]
    # assert str(filter_args[0]) == str(RepositoryPDFLink.repository_id == repo_id)
    # assert str(filter_args[1]) == str(RepositoryPDFLink.branch_name == branch)
    # assert str(filter_args[2]) == str(RepositoryPDFLink.relative_path == rel_path)

    mock_scalar_method.assert_called_once()


@pytest.mark.asyncio
async def test_get_pdf_document_uuid_for_repo_file_not_found(framework_instance_for_pdf_retrieval):
    framework, mock_session = framework_instance_for_pdf_retrieval

    mock_scalar_method = MagicMock(return_value=None) # Simulate not found
    mock_filter_obj = MagicMock()
    mock_filter_obj.scalar_one_or_none = mock_scalar_method
    mock_join_obj = MagicMock()
    mock_join_obj.filter.return_value = mock_filter_obj
    mock_session.query.return_value.join.return_value = mock_join_obj

    result_uuid = await framework.get_pdf_document_uuid_for_repo_file(1, "main", "other.pdf")
    assert result_uuid is None

@pytest.mark.asyncio
async def test_get_pdf_document_uuid_for_repo_file_db_exception(framework_instance_for_pdf_retrieval):
    framework, mock_session = framework_instance_for_pdf_retrieval

    mock_session.query.side_effect = Exception("Simulated DB query failed")

    result_uuid = await framework.get_pdf_document_uuid_for_repo_file(1, "main", "error.pdf")
    assert result_uuid is None
