# tests/ui/test_ui_file_explorer.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import urllib.parse # For checking URL encoded params
import gradio as gr

from sda.ui import DashboardUI
from sda.app import CodeAnalysisFramework
from sda.config import IngestionConfig

MOCK_REPO_ID = 1
MOCK_BRANCH_NAME = "main"
MOCK_REPO_PATH_STR = "/test/mock_repo_root" # Using a more distinct mock path

@pytest.fixture
def mock_framework():
    framework = MagicMock(spec=CodeAnalysisFramework)

    mock_repo = MagicMock()
    mock_repo.path = MOCK_REPO_PATH_STR
    framework.get_repository_by_id.return_value = mock_repo

    # This method is now for retrieving UUID of already processed PDFs
    framework.get_pdf_document_uuid_for_repo_file = AsyncMock()

    # Mocks for code file path
    framework.get_repository_status.return_value = {"modified": [], "new": []}
    framework.get_file_diff_or_content.return_value = ("diff_content_mock_value", None)
    framework.get_file_content_by_path.return_value = "raw_code_content_mock_value"

    return framework

@pytest.fixture
def dashboard_ui_instance(mock_framework):
    iframe_template = '<iframe src="/static/ast_visualization.html?repo_id={repo_id_placeholder}&branch_name={branch_name_placeholder}&file_path={encoded_relative_file_path_placeholder}"></iframe>'
    with patch.object(DashboardUI, '_generate_embedding_html', return_value=iframe_template):
        ui = DashboardUI(framework=mock_framework)
    return ui

@pytest.mark.asyncio
async def test_handle_file_explorer_select_pdf_file_successfully_processed(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    mock_retrieved_pdf_uuid = "retrieved-uuid-for-pdf-123"
    mock_framework.get_pdf_document_uuid_for_repo_file.return_value = mock_retrieved_pdf_uuid

    selected_pdf_abs_path = str(Path(MOCK_REPO_PATH_STR) / "project_docs" / "manual.pdf")
    expected_relative_pdf_path = "project_docs/manual.pdf"

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        repo_id=MOCK_REPO_ID,
        branch=MOCK_BRANCH_NAME,
        selection_event_data=selected_pdf_abs_path
    )

    assert len(outputs) == 5

    embedding_update = outputs[0]
    assert isinstance(embedding_update, dict) and embedding_update.get("__type__") == "update"
    assert f"/static/pdf_visualization.html?pdf_doc_uuid={mock_retrieved_pdf_uuid}" in embedding_update['value']

    code_viewer_update = outputs[1]
    assert code_viewer_update['value'] == "// PDF view active in Embedding tab."
    assert code_viewer_update['visible'] is True

    image_viewer_update = outputs[2]
    assert image_viewer_update['visible'] is False

    assert outputs[3] == expected_relative_pdf_path # selected_file_state

    no_changes_update = outputs[4]
    assert no_changes_update['visible'] is False

    mock_framework.get_pdf_document_uuid_for_repo_file.assert_called_once_with(
        repo_id=MOCK_REPO_ID,
        branch_name=MOCK_BRANCH_NAME,
        relative_path=expected_relative_pdf_path
    )

@pytest.mark.asyncio
async def test_handle_file_explorer_select_code_file_is_clean(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    selected_code_abs_path = str(Path(MOCK_REPO_PATH_STR) / "module" / "service.py")
    expected_relative_code_path = "module/service.py"

    # Ensure this file is "clean" for this test
    mock_framework.get_repository_status.return_value = {"modified": [], "new": []}

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        repo_id=MOCK_REPO_ID,
        branch=MOCK_BRANCH_NAME,
        selection_event_data=selected_code_abs_path
    )

    assert len(outputs) == 5
    embedding_update = outputs[0]
    assert isinstance(embedding_update, dict) and embedding_update.get("__type__") == "update"

    expected_iframe_src = f"/static/ast_visualization.html?repo_id={MOCK_REPO_ID}&branch_name={urllib.parse.quote_plus(MOCK_BRANCH_NAME)}&file_path={urllib.parse.quote_plus(expected_relative_code_path)}"
    assert expected_iframe_src in embedding_update['value']

    mock_framework.get_pdf_document_uuid_for_repo_file.assert_not_called()

    # Raw Diff tab should show "NO CHANGES" for clean code file
    no_changes_update = outputs[4]
    assert no_changes_update['visible'] is True
    code_viewer_update = outputs[1] # Code viewer for diff/content should be hidden
    assert code_viewer_update['visible'] is False
    image_viewer_update = outputs[2]
    assert image_viewer_update['visible'] is False


@pytest.mark.asyncio
async def test_handle_file_explorer_select_pdf_not_yet_processed(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    mock_framework.get_pdf_document_uuid_for_repo_file.return_value = None # Simulate PDF not found in DB

    selected_pdf_abs_path = str(Path(MOCK_REPO_PATH_STR) / "archive" / "old_spec.pdf")

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        MOCK_REPO_ID, MOCK_BRANCH_NAME, selected_pdf_abs_path
    )

    embedding_update = outputs[0]
    assert "PDF not processed or found." in embedding_update['value'] # Check for appropriate message

    code_viewer_update = outputs[1] # Should show error/message
    assert "PDF not processed or found." in code_viewer_update['value']


@pytest.mark.asyncio
async def test_handle_file_explorer_select_raw_image_file(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    selected_img_abs_path = str(Path(MOCK_REPO_PATH_STR) / "images" / "diagram.png")
    expected_relative_img_path = "images/diagram.png"

    mock_pil_image = MagicMock() # Mock PIL.Image object
    with patch('sda.ui.Image.open', return_value=mock_pil_image) as mock_image_open_call:
        outputs = await dashboard_ui_instance.handle_file_explorer_select(
            repo_id=MOCK_REPO_ID,
            branch=MOCK_BRANCH_NAME,
            selection_event_data=selected_img_abs_path
        )

    mock_image_open_call.assert_called_once_with(Path(selected_img_abs_path))
    mock_pil_image.load.assert_called_once() # Check that image data was loaded

    assert len(outputs) == 5
    embedding_update = outputs[0]
    assert "Embedding view not applicable for image" in embedding_update['value']

    code_viewer_update = outputs[1]
    assert code_viewer_update['visible'] is False

    image_viewer_update = outputs[2]
    assert image_viewer_update['value'] == mock_pil_image
    assert image_viewer_update['visible'] is True

    assert outputs[3] == expected_relative_img_path

    no_changes_update = outputs[4]
    assert no_changes_update['visible'] is False
```
