# sda/services/pdf_parser.py
import asyncio
import json
import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from pydantic import BaseModel, Field
from enum import Enum

# --- Pydantic Models (defined in the design step) ---
class PDFElementType(str, Enum):
    PAGE = "page"
    TEXT = "text"
    HEADING = "heading"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"
    LIST_ITEM = "list_item" # Future: if MinerU gives finer list detail
    PARAGRAPH = "paragraph" # Default for text blocks
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    FOOTNOTE = "footnote" # If MinerU separates these well
    UNKNOWN = "unknown" # Fallback

class PDFNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: PDFElementType
    bbox: Optional[List[float]] = None # Relative to page [x0, y0, x1, y1] - MinerU might not give this directly in _content_list.json
    page_number: int
    text_content: Optional[str] = None
    html_content: Optional[str] = None
    image_blob_id: Optional[str] = None
    latex_content: Optional[str] = None
    children: List['PDFNode'] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict) # e.g., MinerU category, text_level for headings

PDFNode.model_rebuild()

class ParsedPDFDocument(BaseModel):
    pdf_file_hash: str
    total_pages: int
    pages: List[PDFNode] = Field(default_factory=list) # List of PAGE type PDFNodes

class PDFImageBlob(BaseModel):
    blob_id: str # Hash of image content
    content_type: str # e.g., "image/png", "image/jpeg"
    data: bytes

# --- MinerU Output Models (approximated from documentation) ---
# These help in parsing MinerU's _content_list.json
class MinerUTextElement(BaseModel):
    type: str # "text"
    text: str
    text_level: Optional[int] = 0
    page_idx: int

class MinerUImageElement(BaseModel):
    type: str # "image"
    img_path: str
    img_caption: Optional[List[str]] = None
    # img_footnote: Optional[List[str]] = None # Not in our immediate plan to use
    page_idx: int

class MinerUTableElement(BaseModel):
    type: str # "table"
    img_path: str # Path to an image of the table
    table_caption: Optional[List[str]] = None
    # table_footnote: Optional[List[str]] = None # Not in our immediate plan to use
    table_body: str # HTML content of the table
    page_idx: int

class MinerUEquationElement(BaseModel):
    type: str # "equation"
    img_path: str # Path to an image of the formula
    text: str # LaTeX content
    # text_format: Optional[str] = "latex" # Not strictly needed if we assume it's LaTeX
    page_idx: int

MinerUElement = Union[MinerUTextElement, MinerUImageElement, MinerUTableElement, MinerUEquationElement]


class PDFParsingService:
    def __init__(self, mineru_path: str = "mineru"): # Allow overriding MinerU path for testing/env
        self.mineru_path = mineru_path
        # TODO: Add logging configuration

    async def _run_mineru(self, pdf_path: Path, output_dir: Path) -> None:
        """Runs the MinerU CLI tool."""
        # Ensure MinerU executable is findable
        mineru_executable = shutil.which(self.mineru_path)
        if not mineru_executable:
            # Try common user local bin if not in PATH
            user_local_bin = Path.home() / ".local" / "bin" / self.mineru_path
            if user_local_bin.exists():
                mineru_executable = str(user_local_bin)
            else:
                raise FileNotFoundError(
                    f"MinerU executable '{self.mineru_path}' not found in PATH or ~/.local/bin/. "
                    "Please ensure MinerU is installed and accessible."
                )

        cmd = [
            mineru_executable,
            "-p", str(pdf_path.resolve()), # Use absolute path for PDF
            "-o", str(output_dir.resolve()), # Use absolute path for output
            "-m", "txt",
            "-b", "pipeline"  # Explicitly select the pipeline backend
        ]

        # print(f"Running MinerU command: {' '.join(cmd)}") # For debugging

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # print(f"MinerU stdout: {stdout.decode(errors='ignore')}") # For debugging
            # print(f"MinerU stderr: {stderr.decode(errors='ignore')}") # For debugging
            raise Exception(f"MinerU failed with error code {process.returncode}: {stderr.decode(errors='ignore')} {stdout.decode(errors='ignore')}")
        # print(f"MinerU finished successfully. stdout: {stdout.decode(errors='ignore')}") # For debugging

    def _calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _map_mineru_element_to_pdfnode(
        self,
        mineru_element: Dict[str, Any],
        mineru_content_dir: Path, # Directory where _content_list.json and images/ are
        image_blobs_map: Dict[str, PDFImageBlob]
    ) -> Optional[PDFNode]:
        node_type_str = mineru_element.get("type")
        page_idx = mineru_element.get("page_idx", -1)

        if page_idx == -1:
            # print(f"Warning: MinerU element missing page_idx: {mineru_element}")
            return None

        node = PDFNode(page_number=page_idx, metadata={"original_mineru_type": node_type_str})

        if node_type_str == "text":
            data = MinerUTextElement(**mineru_element)
            node.text_content = data.text
            text_level = data.text_level or 0
            if text_level > 0:
                node.type = PDFElementType.HEADING
                node.metadata["heading_level"] = text_level
            else:
                node.type = PDFElementType.PARAGRAPH
            node.metadata["text_level"] = text_level

        elif node_type_str == "image":
            data = MinerUImageElement(**mineru_element)
            node.type = PDFElementType.IMAGE

            if data.img_caption:
                node.metadata["caption"] = " ".join(data.img_caption)

            # img_path from MinerU is relative to the dir of _content_list.json
            image_file_path = mineru_content_dir / data.img_path
            if image_file_path.exists():
                try:
                    with open(image_file_path, "rb") as f_img:
                        img_data = f_img.read()
                    img_hash = hashlib.sha256(img_data).hexdigest()

                    if img_hash not in image_blobs_map:
                        ext = image_file_path.suffix.lower()
                        content_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else \
                                       "image/png" if ext == ".png" else \
                                       "image/gif" if ext == ".gif" else \
                                       "application/octet-stream" # Fallback
                        image_blobs_map[img_hash] = PDFImageBlob(blob_id=img_hash, content_type=content_type, data=img_data)

                    node.image_blob_id = img_hash
                except Exception as e:
                    # print(f"Error processing image file {image_file_path}: {e}")
                    return None
            else:
                # print(f"Warning: Image file not found: {image_file_path}")
                return None

        elif node_type_str == "table":
            data = MinerUTableElement(**mineru_element)
            node.type = PDFElementType.TABLE
            node.html_content = data.table_body
            if data.table_caption:
                 node.metadata["caption"] = " ".join(data.table_caption)
            # Optionally handle table image via data.img_path if needed for visualization

        elif node_type_str == "equation":
            data = MinerUEquationElement(**mineru_element)
            node.type = PDFElementType.FORMULA
            node.latex_content = data.text
            # Optionally handle equation image via data.img_path

        else:
            # print(f"Warning: Unknown MinerU element type: {node_type_str}")
            node.type = PDFElementType.UNKNOWN
            node.text_content = json.dumps(mineru_element) # Store raw for inspection

        return node

    async def parse_pdf(self, pdf_file_path_str: str) -> Tuple[ParsedPDFDocument, List[PDFImageBlob]]:
        pdf_file_path = Path(pdf_file_path_str).resolve() # Ensure absolute path
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

        file_hash = self._calculate_file_hash(pdf_file_path)

        # MinerU creates output in a subfolder named after the PDF stem inside the specified output_dir
        # e.g., mineru -p mydoc.pdf -o /tmp/mineru_out -> output is in /tmp/mineru_out/mydoc/
        # So, the TemporaryDirectory itself can be the -o target.
        with tempfile.TemporaryDirectory() as temp_mineru_base_output_dir_str:
            temp_mineru_base_output_dir = Path(temp_mineru_base_output_dir_str)

            await self._run_mineru(pdf_file_path, temp_mineru_base_output_dir)

            pdf_filename_stem = pdf_file_path.stem
            # This is the directory where _content_list.json and images/ folder are expected
            mineru_content_dir = temp_mineru_base_output_dir / pdf_filename_stem
            content_list_json_path = mineru_content_dir / f"{pdf_filename_stem}_content_list.json"

            if not content_list_json_path.exists():
                # Check if MinerU output files directly into the -o path (less common but possible)
                content_list_json_path_alt = temp_mineru_base_output_dir / f"{pdf_filename_stem}_content_list.json"
                if content_list_json_path_alt.exists():
                    content_list_json_path = content_list_json_path_alt
                    mineru_content_dir = temp_mineru_base_output_dir # Adjust content dir if files are flat
                else:
                    raise FileNotFoundError(
                        f"MinerU output _content_list.json not found. Looked in {mineru_content_dir} "
                        f"and {temp_mineru_base_output_dir}."
                    )

            if not mineru_content_dir.is_dir():
                 raise NotADirectoryError(f"MinerU content directory not found or is not a directory: {mineru_content_dir}")


            with open(content_list_json_path, "r", encoding="utf-8") as f:
                mineru_data_list = json.load(f)

            parsed_doc = ParsedPDFDocument(pdf_file_hash=file_hash, total_pages=0)
            page_children_map: Dict[int, List[PDFNode]] = {}
            max_page_num = -1
            image_blobs_map: Dict[str, PDFImageBlob] = {}

            for mineru_element_dict in mineru_data_list:
                pdf_node = self._map_mineru_element_to_pdfnode(mineru_element_dict, mineru_content_dir, image_blobs_map)
                if pdf_node:
                    page_num = pdf_node.page_number
                    if page_num not in page_children_map:
                        page_children_map[page_num] = []
                    page_children_map[page_num].append(pdf_node)
                    if page_num > max_page_num:
                        max_page_num = page_num

            parsed_doc.total_pages = max_page_num + 1

            for page_num in sorted(page_children_map.keys()):
                page_node = PDFNode(
                    type=PDFElementType.PAGE,
                    page_number=page_num,
                    children=page_children_map[page_num]
                )
                parsed_doc.pages.append(page_node)

            return parsed_doc, list(image_blobs_map.values())

# Example Usage (commented out for non-direct execution)
# async def main_test():
#     # Ensure MinerU is installed and in PATH, or PDFParsingService(mineru_path="/path/to/mineru")
#     # Create a dummy PDF:
#     # from reportlab.pdfgen import canvas
#     # def create_dummy_pdf(filename="dummy_test.pdf"):
#     #     c = canvas.Canvas(filename)
#     #     c.drawString(100, 750, "Hello World - Page 1")
#     #     c.showPage()
#     #     c.drawString(100, 750, "This is Page 2.")
#     #     # You might need to add an image to the PDF for full testing of image extraction
#     #     c.save()
#     # if not Path("dummy_test.pdf").exists():
#     #    create_dummy_pdf()
#
#     parser = PDFParsingService() # Or PDFParsingService(mineru_path="~/.local/bin/mineru")
#     pdf_to_parse = "dummy_test.pdf" # Replace with a real PDF for actual testing
#
#     if not Path(pdf_to_parse).exists():
#         print(f"Test PDF {pdf_to_parse} not found. Please create it or specify another PDF.")
#         return
#
#     print(f"Attempting to parse {pdf_to_parse}...")
#     try:
#         parsed_document, image_blobs = await parser.parse_pdf(pdf_to_parse)
#         print("\nParsed Document Structure:")
#         # print(parsed_document.model_dump_json(indent=2)) # Can be very verbose
#         print(f"PDF Hash: {parsed_document.pdf_file_hash}")
#         print(f"Total Pages: {parsed_document.total_pages}")
#         for page in parsed_document.pages:
#             print(f"  Page {page.page_number + 1}: {len(page.children)} elements")
#
#         print(f"\nExtracted {len(image_blobs)} unique image blobs.")
#         for blob in image_blobs:
#             print(f"  Blob ID: {blob.blob_id}, Content Type: {blob.content_type}, Size: {len(blob.data)} bytes")
#
#     except Exception as e:
#         import traceback
#         print(f"Error during PDF parsing test: {e}")
#         print(traceback.format_exc())
#
# if __name__ == "__main__":
#     # asyncio.run(main_test())
#     print("PDFParsingService defined. To test, call main_test() after ensuring MinerU and a test PDF are available.")
