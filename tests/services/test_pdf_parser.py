# tests/services/test_pdf_parser.py
import asyncio
import json
import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union # Added Union
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Models from the service itself
from sda.services.pdf_parser import (
    PDFParsingService,
    ParsedPDFDocument,
    PDFImageBlob,
    PDFElementType,
    PDFNode
)

# Sample MinerU _content_list.json data
SAMPLE_MINERU_CONTENT_LIST_JSON = [
    {
        "type": "text",
        "text": "Hello World",
        "text_level": 1, # Heading 1
        "page_idx": 0
    },
    {
        "type": "text",
        "text": "This is a paragraph.",
        "text_level": 0,
        "page_idx": 0
    },
    {
        "type": "image",
        "img_path": "images/image1.png", # Relative to the JSON file's directory
        "img_caption": ["This is an image caption."],
        "page_idx": 0
    },
    {
        "type": "table",
        "img_path": "images/table1.png", # Image of the table
        "table_caption": ["This is a table caption."],
        "table_body": "<table><tr><td>Data</td></tr></table>",
        "page_idx": 1
    },
    {
        "type": "equation",
        "img_path": "images/formula1.png", # Image of the formula
        "text": "E = mc^2", # LaTeX
        "page_idx": 1
    }
]

# Dummy image data for mocking
DUMMY_PNG_DATA = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
DUMMY_JPG_DATA = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x11\x11\x18!\x1e\x18\x1a\x1d(%\x18\x1c#\x1c%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9'


@pytest.fixture
def mock_mineru_output_base_dir():
    """
    Creates a temporary directory that acts as the base output directory specified to MinerU's -o flag.
    MinerU is expected to create a subdirectory named after the PDF stem within this base directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        yield Path(temp_dir_str)


@pytest.fixture
def setup_mineru_output_files(mock_mineru_output_base_dir: Path):
    """
    Sets up the actual content files (JSON, images) inside a subdirectory
    within the mock_mineru_output_base_dir.
    This fixture depends on mock_mineru_output_base_dir.
    """
    pdf_stem = "test_pdf" # Assumed PDF stem name for testing
    mineru_content_actual_dir = mock_mineru_output_base_dir / pdf_stem
    mineru_content_actual_dir.mkdir(parents=True, exist_ok=True)

    content_list_path = mineru_content_actual_dir / f"{pdf_stem}_content_list.json"
    with open(content_list_path, "w") as f:
        json.dump(SAMPLE_MINERU_CONTENT_LIST_JSON, f)

    images_dir = mineru_content_actual_dir / "images"
    images_dir.mkdir(exist_ok=True)
    with open(images_dir / "image1.png", "wb") as f:
        f.write(DUMMY_PNG_DATA)
    with open(images_dir / "table1.png", "wb") as f:
        f.write(DUMMY_PNG_DATA)
    with open(images_dir / "formula1.png", "wb") as f:
        f.write(DUMMY_PNG_DATA)

    return mock_mineru_output_base_dir # Return the base dir, service needs to find content within


@pytest.mark.asyncio
@patch('sda.services.pdf_parser.shutil.which')
@patch('sda.services.pdf_parser.asyncio.create_subprocess_exec')
async def test_parse_pdf_success(mock_create_subprocess_exec, mock_shutil_which, setup_mineru_output_files, tmp_path):
    mock_shutil_which.return_value = "mineru" # Assume mineru is in PATH for the mock

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"MinerU output", b""))
    mock_process.returncode = 0
    mock_create_subprocess_exec.return_value = mock_process

    # Use the 'setup_mineru_output_files' fixture which gives the base output dir
    mineru_base_output_dir = setup_mineru_output_files

    # Create a dummy PDF file. The service will calculate its hash.
    # The PDF file itself doesn't need to be in the mineru_base_output_dir.
    # It can be anywhere, like tmp_path for this test.
    dummy_pdf_file = tmp_path / "test_pdf.pdf" # Name matches assumed stem in setup_mineru_output_files
    dummy_pdf_content = b"dummy pdf content for test_parse_pdf_success"
    with open(dummy_pdf_file, "wb") as f:
        f.write(dummy_pdf_content)

    parser = PDFParsingService()
    # Pass the path to the dummy PDF and expect MinerU to write to mineru_base_output_dir
    parsed_doc, image_blobs = await parser.parse_pdf(str(dummy_pdf_file))

    assert mock_create_subprocess_exec.called
    args, _ = mock_create_subprocess_exec.call_args
    assert args[0] == "mineru"
    assert args[2] == str(dummy_pdf_file.resolve())
    assert args[4] == str(mineru_base_output_dir.resolve())
    assert args[6] == "txt"

    assert parsed_doc is not None
    assert parsed_doc.pdf_file_hash == hashlib.sha256(dummy_pdf_content).hexdigest()
    assert parsed_doc.total_pages == 2
    assert len(parsed_doc.pages) == 2

    page0_nodes = parsed_doc.pages[0].children
    assert len(page0_nodes) == 3
    assert page0_nodes[0].type == PDFElementType.HEADING
    assert page0_nodes[0].text_content == "Hello World"
    assert page0_nodes[0].metadata["heading_level"] == 1
    assert page0_nodes[1].type == PDFElementType.PARAGRAPH
    assert page0_nodes[1].text_content == "This is a paragraph."
    assert page0_nodes[2].type == PDFElementType.IMAGE
    assert page0_nodes[2].image_blob_id is not None
    assert page0_nodes[2].metadata["caption"] == "This is an image caption."

    page1_nodes = parsed_doc.pages[1].children
    assert len(page1_nodes) == 2
    assert page1_nodes[0].type == PDFElementType.TABLE
    assert page1_nodes[0].html_content == "<table><tr><td>Data</td></tr></table>"
    assert page1_nodes[0].metadata["caption"] == "This is a table caption."
    assert page1_nodes[1].type == PDFElementType.FORMULA
    assert page1_nodes[1].latex_content == "E = mc^2"

    assert len(image_blobs) == 1
    image1_hash = hashlib.sha256(DUMMY_PNG_DATA).hexdigest()
    assert image_blobs[0].blob_id == image1_hash
    assert image_blobs[0].content_type == "image/png"
    assert image_blobs[0].data == DUMMY_PNG_DATA
    assert page0_nodes[2].image_blob_id == image1_hash


@pytest.mark.asyncio
@patch('sda.services.pdf_parser.shutil.which')
@patch('sda.services.pdf_parser.asyncio.create_subprocess_exec')
async def test_parse_pdf_mineru_failure(mock_create_subprocess_exec, mock_shutil_which, tmp_path):
    mock_shutil_which.return_value = "mineru"
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"MinerU critical error"))
    mock_process.returncode = 1
    mock_create_subprocess_exec.return_value = mock_process

    dummy_pdf_path = tmp_path / "fail_test.pdf"
    with open(dummy_pdf_path, "wb") as f:
        f.write(b"content")

    parser = PDFParsingService()
    with pytest.raises(Exception, match="MinerU failed with error code 1: MinerU critical error "):
        await parser.parse_pdf(str(dummy_pdf_path))

def test_map_mineru_text_element():
    parser = PDFParsingService()
    mineru_elem = {"type": "text", "text": "Sample Text", "text_level": 0, "page_idx": 0}
    node = parser._map_mineru_element_to_pdfnode(mineru_elem, Path("."), {})
    assert node.type == PDFElementType.PARAGRAPH
    assert node.text_content == "Sample Text"
    assert node.page_number == 0

def test_map_mineru_heading_element():
    parser = PDFParsingService()
    mineru_elem = {"type": "text", "text": "Main Title", "text_level": 2, "page_idx": 1}
    node = parser._map_mineru_element_to_pdfnode(mineru_elem, Path("."), {})
    assert node.type == PDFElementType.HEADING
    assert node.text_content == "Main Title"
    assert node.metadata["heading_level"] == 2
    assert node.page_number == 1

def test_map_mineru_image_element_with_file(tmp_path):
    parser = PDFParsingService()
    # tmp_path here represents the 'mineru_content_dir' which is where _content_list.json would be
    # and where 'images/' subdir is expected.
    images_subdir = tmp_path / "images"
    images_subdir.mkdir()
    img_file_path_in_images_subdir = images_subdir / "test_image.jpg"
    with open(img_file_path_in_images_subdir, "wb") as f:
        f.write(DUMMY_JPG_DATA)

    mineru_elem = {
        "type": "image",
        "img_path": "images/test_image.jpg", # This path is relative to 'mineru_content_dir' (tmp_path)
        "page_idx": 0
    }
    image_blobs_map = {}
    node = parser._map_mineru_element_to_pdfnode(mineru_elem, tmp_path, image_blobs_map) # Pass tmp_path as mineru_content_dir

    assert node is not None
    assert node.type == PDFElementType.IMAGE
    assert node.image_blob_id is not None
    expected_hash = hashlib.sha256(DUMMY_JPG_DATA).hexdigest()
    assert node.image_blob_id == expected_hash
    assert expected_hash in image_blobs_map
    assert image_blobs_map[expected_hash].content_type == "image/jpeg"
    assert image_blobs_map[expected_hash].data == DUMMY_JPG_DATA

def test_map_mineru_image_element_file_not_found(tmp_path):
    parser = PDFParsingService()
    mineru_elem = {
        "type": "image",
        "img_path": "images/nonexistent.png", # Relative to tmp_path
        "page_idx": 0
    }
    image_blobs_map = {}
    node = parser._map_mineru_element_to_pdfnode(mineru_elem, tmp_path, image_blobs_map)
    assert node is None
    assert not image_blobs_map
```
