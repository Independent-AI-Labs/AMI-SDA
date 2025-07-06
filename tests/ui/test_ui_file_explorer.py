# tests/ui/test_ui_file_explorer.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import urllib.parse
import gradio as gr # For gr.update and gr.skip

from sda.ui import DashboardUI
from sda.app import CodeAnalysisFramework
from sda.config import IngestionConfig

MOCK_REPO_ID = 1
MOCK_BRANCH_NAME = "main"
MOCK_REPO_PATH_STR = "/test/mock_repo" # Use a string for consistency with Path handling

@pytest.fixture
def mock_framework():
    framework = MagicMock(spec=CodeAnalysisFramework)

    mock_repo = MagicMock()
    mock_repo.path = MOCK_REPO_PATH_STR
    framework.get_repository_by_id.return_value = mock_repo

    framework.get_or_process_pdf_document = AsyncMock()

    framework.get_repository_status.return_value = {"modified": [], "new": []}
    framework.get_file_diff_or_content.return_value = ("diff_content_mock", None)
    framework.get_file_content_by_path.return_value = "raw_code_content_mock"

    return framework

@pytest.fixture
def dashboard_ui_instance(mock_framework):
    # Patch _generate_embedding_html as its exact output for code files isn't the primary focus here,
    # rather that the correct logic path is taken.
    # The template string should contain placeholders that handle_file_explorer_select replaces.
    iframe_template = '<iframe src="/static/ast_visualization.html?repo_id={repo_id_placeholder}&branch_name={branch_name_placeholder}&file_path={encoded_relative_file_path_placeholder}"></iframe>'
    with patch.object(DashboardUI, '_generate_embedding_html', return_value=iframe_template):
        ui = DashboardUI(framework=mock_framework)
    return ui

@pytest.mark.asyncio
async def test_handle_file_explorer_select_pdf_file(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    mock_pdf_uuid = "test-pdf-doc-uuid-xyz"
    mock_framework.get_or_process_pdf_document.return_value = mock_pdf_uuid

    selected_pdf_abs_path = str(Path(MOCK_REPO_PATH_STR) / "documents" / "report.pdf")

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        repo_id=MOCK_REPO_ID,
        branch=MOCK_BRANCH_NAME,
        selection_event_data=selected_pdf_abs_path
    )

    assert len(outputs) == 5

    embedding_update = outputs[0]
    assert isinstance(embedding_update, dict) and embedding_update.get("__type__") == "update"
    assert f"/static/pdf_visualization.html?pdf_doc_uuid={mock_pdf_uuid}" in embedding_update['value']

    code_viewer_update = outputs[1]
    assert isinstance(code_viewer_update, dict) and code_viewer_update.get("__type__") == "update"
    assert code_viewer_update['value'] == "// PDF view active in Embedding tab."
    assert code_viewer_update['visible'] is True

    image_viewer_update = outputs[2]
    assert isinstance(image_viewer_update, dict) and image_viewer_update.get("__type__") == "update"
    assert image_viewer_update['visible'] is False

    assert outputs[3] == "documents/report.pdf" # selected_file_state (relative path)

    no_changes_update = outputs[4]
    assert isinstance(no_changes_update, dict) and no_changes_update.get("__type__") == "update"
    assert no_changes_update['visible'] is False

    mock_framework.get_or_process_pdf_document.assert_called_once_with(
        absolute_pdf_path=selected_pdf_abs_path,
        repo_id=MOCK_REPO_ID,
        branch_name=MOCK_BRANCH_NAME,
        relative_path="documents/report.pdf"
    )

@pytest.mark.asyncio
async def test_handle_file_explorer_select_code_file(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    selected_code_abs_path = str(Path(MOCK_REPO_PATH_STR) / "src" / "main.py")

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        repo_id=MOCK_REPO_ID,
        branch=MOCK_BRANCH_NAME,
        selection_event_data=selected_code_abs_path
    )

    assert len(outputs) == 5
    embedding_update = outputs[0]
    assert isinstance(embedding_update, dict) and embedding_update.get("__type__") == "update"

    expected_iframe_src = f"/static/ast_visualization.html?repo_id={MOCK_REPO_ID}&branch_name={urllib.parse.quote_plus(MOCK_BRANCH_NAME)}&file_path={urllib.parse.quote_plus('src/main.py')}"
    assert expected_iframe_src in embedding_update['value']

    mock_framework.get_or_process_pdf_document.assert_not_called()
    # Raw Diff tab should show "NO CHANGES" as default status is clean
    no_changes_update = outputs[4]
    assert no_changes_update['visible'] is True
    code_viewer_update = outputs[1]
    assert code_viewer_update['visible'] is False


@pytest.mark.asyncio
async def test_handle_file_explorer_select_pdf_processing_fails(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    mock_framework.get_or_process_pdf_document.return_value = None # Simulate failure

    selected_pdf_abs_path = str(Path(MOCK_REPO_PATH_STR) / "docs" / "failing.pdf")

    outputs = await dashboard_ui_instance.handle_file_explorer_select(
        MOCK_REPO_ID, MOCK_BRANCH_NAME, selected_pdf_abs_path
    )

    embedding_update = outputs[0]
    assert "Could not get or process PDF" in embedding_update['value']

    code_viewer_update = outputs[1]
    assert "Could not get or process PDF" in code_viewer_update['value']

@pytest.mark.asyncio
async def test_handle_file_explorer_select_raw_image_file(dashboard_ui_instance: DashboardUI, mock_framework: MagicMock):
    selected_img_abs_path = str(Path(MOCK_REPO_PATH_STR) / "assets" / "logo.png")

    # Mock Image.open
    mock_pil_image = MagicMock()
    with patch('sda.ui.Image.open', return_value=mock_pil_image) as mock_image_open:
        outputs = await dashboard_ui_instance.handle_file_explorer_select(
            repo_id=MOCK_REPO_ID,
            branch=MOCK_BRANCH_NAME,
            selection_event_data=selected_img_abs_path
        )

    mock_image_open.assert_called_once_with(Path(selected_img_abs_path))
    mock_pil_image.load.assert_called_once()

    assert len(outputs) == 5
    embedding_update = outputs[0] # Embedding view message
    assert "Embedding view not applicable for image" in embedding_update['value']

    code_viewer_update = outputs[1] # Code viewer hidden
    assert code_viewer_update['visible'] is False

    image_viewer_update = outputs[2] # Image viewer shows image
    assert image_viewer_update['value'] == mock_pil_image
    assert image_viewer_update['visible'] is True

    assert outputs[3] == "assets/logo.png" # Relative path for state

    no_changes_update = outputs[4] # No changes message hidden
    assert no_changes_update['visible'] is False
```
