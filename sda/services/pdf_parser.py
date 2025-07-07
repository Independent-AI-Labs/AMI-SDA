# sda/services/pdf_parser.py
import asyncio
import json
import logging
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

        # Get configured parsing method, default to "auto" if invalid
        from sda.config import IngestionConfig # Local import to avoid circular dependency at module level if any
        configured_method = IngestionConfig.MINERU_PDF_PARSE_METHOD.lower()
        if configured_method not in ["auto", "txt", "ocr"]:
            logging.warning(
                f"Invalid MINERU_PDF_PARSE_METHOD '{configured_method}' in config. "
                f"Defaulting to 'auto'. Valid options are 'auto', 'txt', 'ocr'."
            )
            parse_method = "auto"
        else:
            parse_method = configured_method

        cmd = [
            mineru_executable,
            "-p", str(pdf_path.resolve()), # Use absolute path for PDF or directory
            "-o", str(output_dir.resolve()), # Use absolute path for output
            "-m", parse_method, # Use configured method
            "-b", "pipeline"  # Explicitly select the pipeline backend
        ]

        # print(f"Running MinerU command: {' '.join(cmd)}") # For debugging
        pdf_filename = pdf_path.name # For clearer logging
        log_prefix = f"[MinerU Runner File:{pdf_filename}]"
        logging.info(f"{log_prefix} Executing MinerU command: {' '.join(cmd)}")

        all_stdout_lines = []
        all_stderr_lines = []

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream stdout
        async def stream_stdout():
            while process.stdout and not process.stdout.at_eof():
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors='ignore').strip()
                if line:
                    all_stdout_lines.append(line)
                    logging.info(f"{log_prefix} STDOUT: {line}")

        # Stream stderr
        async def stream_stderr():
            while process.stderr and not process.stderr.at_eof():
                line_bytes = await process.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors='ignore').strip()
                if line:
                    all_stderr_lines.append(line)
                    logging.warning(f"{log_prefix} STDERR: {line}") # Log stderr as warning

        # Run streaming tasks concurrently
        await asyncio.gather(stream_stdout(), stream_stderr())

        # Wait for the process to complete
        await process.wait()

        returncode = process.returncode
        stdout_full = "\n".join(all_stdout_lines)
        stderr_full = "\n".join(all_stderr_lines)

        logging.info(f"{log_prefix} MinerU command finished with return code {returncode}.")

        if returncode != 0:
            # The error will be raised by the caller based on this return dict
            logging.error(f"{log_prefix} MinerU failed. Full stdout and stderr captured.")

        return {
            "stdout": stdout_full,
            "stderr": stderr_full,
            "returncode": returncode,
            "command": ' '.join(cmd)
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _map_mineru_element_to_pdfnode(
from io import BytesIO # For in-memory image manipulation
from PIL import Image # For image processing

from sda.config import IngestionConfig # For image processing settings


class PDFParsingService:
    def __init__(self, mineru_path: str = "mineru"): # Allow overriding MinerU path for testing/env
        self.mineru_path = mineru_path
        # TODO: Add logging configuration

    async def _run_mineru(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
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

        # Get configured parsing method, default to "auto" if invalid
        # from sda.config import InestionConfig # Already imported at class level
        configured_method = IngestionConfig.MINERU_PDF_PARSE_METHOD.lower()
        if configured_method not in ["auto", "txt", "ocr"]:
            logging.warning(
                f"Invalid MINERU_PDF_PARSE_METHOD '{configured_method}' in config. "
                f"Defaulting to 'auto'. Valid options are 'auto', 'txt', 'ocr'."
            )
            parse_method = "auto"
        else:
            parse_method = configured_method

        cmd = [
            mineru_executable,
            "-p", str(pdf_path.resolve()), # Use absolute path for PDF or directory
            "-o", str(output_dir.resolve()), # Use absolute path for output
            "-m", parse_method, # Use configured method
            "-b", "pipeline"  # Explicitly select the pipeline backend
        ]

        # print(f"Running MinerU command: {' '.join(cmd)}") # For debugging
        pdf_filename = pdf_path.name # For clearer logging
        log_prefix = f"[MinerU Runner File:{pdf_filename}]"
        logging.info(f"{log_prefix} Executing MinerU command: {' '.join(cmd)}")

        all_stdout_lines = []
        all_stderr_lines = []

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream stdout
        async def stream_stdout():
            while process.stdout and not process.stdout.at_eof():
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors='ignore').strip()
                if line:
                    all_stdout_lines.append(line)
                    logging.info(f"{log_prefix} STDOUT: {line}")

        # Stream stderr
        async def stream_stderr():
            while process.stderr and not process.stderr.at_eof():
                line_bytes = await process.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors='ignore').strip()
                if line:
                    all_stderr_lines.append(line)
                    logging.warning(f"{log_prefix} STDERR: {line}") # Log stderr as warning

        # Run streaming tasks concurrently
        await asyncio.gather(stream_stdout(), stream_stderr())

        # Wait for the process to complete
        await process.wait()

        returncode = process.returncode
        stdout_full = "\n".join(all_stdout_lines)
        stderr_full = "\n".join(all_stderr_lines)

        logging.info(f"{log_prefix} MinerU command finished with return code {returncode}.")

        if returncode != 0:
            # The error will be raised by the caller based on this return dict
            logging.error(f"{log_prefix} MinerU failed. Full stdout and stderr captured.")

        return {
            "stdout": stdout_full,
            "stderr": stderr_full,
            "returncode": returncode,
            "command": ' '.join(cmd)
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _calculate_bytes_hash(self, data_bytes: bytes) -> str:
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
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

            image_file_path = mineru_content_dir / data.img_path
            if image_file_path.exists():
                try:
                    with open(image_file_path, "rb") as f_img_orig:
                        original_image_bytes = f_img_orig.read()

                    img = Image.open(BytesIO(original_image_bytes))

                    # Resize if necessary
                    max_dim = IngestionConfig.MAX_IMAGE_DIMENSION_PX
                    if max_dim and max_dim > 0 and (img.width > max_dim or img.height > max_dim):
                        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        logging.info(f"Resized image {data.img_path} to fit within {max_dim}x{max_dim} -> new size {img.width}x{img.height}")

                    # Convert to PNG
                    output_image_bytes_io = BytesIO()
                    img.save(output_image_bytes_io, format=IngestionConfig.IMAGE_OUTPUT_FORMAT) # Should be "PNG"
                    processed_image_bytes = output_image_bytes_io.getvalue()

                    # Calculate hash of the processed (resized and PNG converted) image bytes
                    img_hash = self._calculate_bytes_hash(processed_image_bytes)

                    if img_hash not in image_blobs_map:
                        image_blobs_map[img_hash] = PDFImageBlob(
                            blob_id=img_hash,
                            content_type=f"image/{IngestionConfig.IMAGE_OUTPUT_FORMAT.lower()}", # e.g., "image/png"
                            data=processed_image_bytes
                        )
                    node.image_blob_id = img_hash
                except Exception as e:
                    logging.error(f"Error processing image file {image_file_path} for node type '{node_type_str}': {e}", exc_info=True)
                    # Decide if this is fatal for the node or if we skip the image part
                    # For now, let's make the node but without image_blob_id if processing failed
                    node.image_blob_id = None
                    # Optionally, could return None to skip the whole PDFNode if image is critical
                    # return None
            else:
                logging.warning(f"Image file not found: {image_file_path} for node type '{node_type_str}'")
                node.image_blob_id = None # Or return None

        elif node_type_str == "table":
            # TODO: Future - if tables also have images that need standardization (data.img_path for table image)
            # The current MinerUTableElement model has img_path for an image of the table.
            # If this needs to be stored and standardized, similar logic as "image" type would apply here.
            # For now, only processing `type: "image"` for standardization.
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

    def _parse_mineru_content_list_json(
        self,
        content_list_json_path: Path,
        mineru_content_dir_for_images: Path, # Base directory for resolving relative image paths in JSON
        file_hash: str, # Pre-calculated hash of the original PDF
        image_blobs_map: Dict[str, PDFImageBlob] # Shared map to accumulate unique image blobs
    ) -> Optional[ParsedPDFDocument]:
        """
        Parses a single _content_list.json file from MinerU output.
        Updates the shared image_blobs_map.
        Returns a ParsedPDFDocument or None if parsing fails.
        """
        log_prefix = f"[PDFParser JSON:{content_list_json_path.name}]"
        if not content_list_json_path.exists():
            logging.warning(f"{log_prefix} _content_list.json not found at {content_list_json_path}")
            return None

        if not mineru_content_dir_for_images.is_dir():
            logging.warning(f"{log_prefix} MinerU content directory for images ({mineru_content_dir_for_images}) is not a directory. Image paths might be incorrect.")
            # Continue parsing JSON, image loading might fail later.

        logging.info(f"{log_prefix} Parsing from: {content_list_json_path}, images relative to: {mineru_content_dir_for_images}")

        try:
            with open(content_list_json_path, "r", encoding="utf-8") as f:
                mineru_data_list = json.load(f)
        except Exception as e:
            logging.error(f"{log_prefix} Failed to read or parse JSON file {content_list_json_path}: {e}", exc_info=True)
            return None

        parsed_doc = ParsedPDFDocument(pdf_file_hash=file_hash, total_pages=0)
        page_children_map: Dict[int, List[PDFNode]] = {}
        max_page_num = -1

        for mineru_element_dict in mineru_data_list:
            pdf_node = self._map_mineru_element_to_pdfnode(mineru_element_dict, mineru_content_dir_for_images, image_blobs_map)
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

        return parsed_doc

    async def parse_single_pdf(self, pdf_file_path_str: str) -> Tuple[Optional[ParsedPDFDocument], List[PDFImageBlob]]:
        """
        Parses a single PDF file using MinerU.
        This is the original parse_pdf method, renamed for clarity.
        """
        pdf_file_path = Path(pdf_file_path_str).resolve()
        if not pdf_file_path.exists():
            logging.error(f"PDF file not found: {pdf_file_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

        file_hash = self._calculate_file_hash(pdf_file_path)
        image_blobs_map: Dict[str, PDFImageBlob] = {} # Specific to this single PDF call

        with tempfile.TemporaryDirectory() as temp_mineru_base_output_dir_str:
            temp_mineru_base_output_dir = Path(temp_mineru_base_output_dir_str)
            pdf_filename_stem = pdf_file_path.stem
            log_prefix = f"[PDFParser SingleFile:{pdf_filename_stem}]"

            mineru_result = await self._run_mineru(pdf_file_path, temp_mineru_base_output_dir)

            if mineru_result["returncode"] != 0:
                error_message = (
                    f"{log_prefix} MinerU execution failed with code {mineru_result['returncode']}.\n"
                    f"Command: {mineru_result['command']}\n"
                    f"MinerU STDOUT:\n{mineru_result['stdout']}\n"
                    f"MinerU STDERR:\n{mineru_result['stderr']}"
                )
                logging.error(error_message)
                # Consider not raising RuntimeError here directly, but returning None,
                # so the caller in workers.py can log it appropriately for that file.
                # For now, keep the raise to match original behavior for single parse.
                raise RuntimeError(f"MinerU failed for {pdf_filename_stem}. STDERR: {mineru_result['stderr'][:500]}")

            logging.info(f"{log_prefix} MinerU execution successful.")

            # Determine path to _content_list.json and image base directory
            # MinerU creates a subdir named after the PDF stem in the output dir
            expected_mineru_output_subdir = temp_mineru_base_output_dir / pdf_filename_stem
            content_list_json_path = expected_mineru_output_subdir / f"{pdf_filename_stem}_content_list.json"
            mineru_content_dir_for_images = expected_mineru_output_subdir

            if not content_list_json_path.exists():
                # Fallback: Check if MinerU output files directly into the base output directory (older behavior or simple cases)
                content_list_json_path_alt = temp_mineru_base_output_dir / f"{pdf_filename_stem}_content_list.json"
                if content_list_json_path_alt.exists():
                    content_list_json_path = content_list_json_path_alt
                    mineru_content_dir_for_images = temp_mineru_base_output_dir
                else:
                    logging.error(f"{log_prefix} MinerU output _content_list.json not found at primary or fallback paths.")
                    # Add directory listing for debugging
                    logging.info(f"Debug: Listing contents of MinerU base output directory ({temp_mineru_base_output_dir}):")
                    for item in temp_mineru_base_output_dir.iterdir():
                        logging.info(f"  - {item.name}{'/' if item.is_dir() else ''}")
                        if item.is_dir():
                            for sub_item in item.iterdir(): logging.info(f"    - {sub_item.name}")
                    return None, [] # Return None if JSON not found

            parsed_doc = self._parse_mineru_content_list_json(
                content_list_json_path,
                mineru_content_dir_for_images,
                file_hash,
                image_blobs_map
            )
            return parsed_doc, list(image_blobs_map.values())

    async def parse_pdfs_from_directory_input(
        self,
        original_pdf_paths: List[Path], # List of absolute paths to the original PDFs that were symlinked
        mineru_input_dir: Path, # The temporary directory containing symlinks, passed to MinerU's -p
    ) -> Tuple[List[Dict[str, Any]], List[PDFImageBlob]]:
        """
        Processes a directory of PDFs (symlinks) using a single MinerU CLI call.
        Returns a list of dictionaries, each representing the outcome for an original PDF,
        and a consolidated list of unique image blobs.
        Each dictionary in the list will have:
        {
            "original_path": str,
            "document": Optional[ParsedPDFDocument],
            "status": "success" | "json_not_found" | "json_parse_error",
            "error_message": Optional[str]
        }
        """
        results: List[Dict[str, Any]] = []
        all_image_blobs_map: Dict[str, PDFImageBlob] = {}
        log_prefix = f"[PDFParser BatchDir:{mineru_input_dir.name}]"
        mineru_batch_stderr = "" # To store stderr from the batch MinerU call

        with tempfile.TemporaryDirectory() as temp_mineru_base_output_dir_str:
            temp_mineru_base_output_dir = Path(temp_mineru_base_output_dir_str)
            logging.info(f"{log_prefix} Running MinerU on directory: {mineru_input_dir} -> output to: {temp_mineru_base_output_dir}")

            mineru_cli_result = await self._run_mineru(mineru_input_dir, temp_mineru_base_output_dir)
            mineru_batch_stderr = mineru_cli_result.get("stderr", "")

            if mineru_cli_result["returncode"] != 0:
                logging.warning(
                    f"{log_prefix} MinerU execution for directory returned code {mineru_cli_result['returncode']}.\n"
                    f"Command: {mineru_cli_result['command']}\n"
                    f"Attempting to parse any successful outputs. STDERR snippet: {mineru_batch_stderr[:500]}"
                )

            for original_pdf_path in original_pdf_paths:
                original_pdf_path_str = str(original_pdf_path.resolve())
                pdf_filename_stem = original_pdf_path.stem

                current_pdf_result = {
                    "original_path": original_pdf_path_str,
                    "document": None,
                    "status": "", # Will be set below
                    "error_message": None
                }

                expected_pdf_output_subdir = temp_mineru_base_output_dir / pdf_filename_stem
                content_list_json_path = expected_pdf_output_subdir / f"{pdf_filename_stem}_content_list.json"
                mineru_content_dir_for_images = expected_pdf_output_subdir

                logging.info(f"{log_prefix} Attempting to parse result for: {original_pdf_path.name} from {content_list_json_path}")

                if content_list_json_path.exists():
                    file_hash = self._calculate_file_hash(original_pdf_path)
                    parsed_doc = self._parse_mineru_content_list_json(
                        content_list_json_path,
                        mineru_content_dir_for_images,
                        file_hash,
                        all_image_blobs_map
                    )
                    if parsed_doc:
                        current_pdf_result["document"] = parsed_doc
                        current_pdf_result["status"] = "success"
                        logging.info(f"{log_prefix} Successfully parsed: {original_pdf_path.name}")
                    else:
                        current_pdf_result["status"] = "json_parse_error"
                        current_pdf_result["error_message"] = f"Failed to parse content_list.json for: {original_pdf_path.name}"
                        logging.warning(f"{log_prefix} {current_pdf_result['error_message']}")
                else:
                    current_pdf_result["status"] = "json_not_found"
                    current_pdf_result["error_message"] = (
                        f"Output JSON not found for: {original_pdf_path.name}. "
                        "Assuming MinerU failed or skipped it."
                    )
                    # Check if general MinerU stderr gives a clue (this is a rough check)
                    if pdf_filename_stem in mineru_batch_stderr or original_pdf_path.name in mineru_batch_stderr:
                         current_pdf_result["error_message"] += f" MinerU stderr might contain related error: {mineru_batch_stderr[:200]}"
                    logging.warning(f"{log_prefix} {current_pdf_result['error_message']}")

                results.append(current_pdf_result)

        return results, list(all_image_blobs_map.values())


# Example Usage (commented out for non-direct execution)
# async def main_test_single():
#     parser = PDFParsingService()
#     # ... (setup for single PDF test)
#     parsed_document, image_blobs = await parser.parse_single_pdf("path_to_your.pdf")
#     # ... (print results)

# async def main_test_batch():
#     parser = PDFParsingService()
#     # Create a temp dir with some symlinked PDFs for testing
#     # For example:
#     # with tempfile.TemporaryDirectory() as input_dir_for_mineru:
#     #     input_dir_path = Path(input_dir_for_mineru)
#     #     pdf_files_to_link = [Path("pdf1.pdf"), Path("pdf2.pdf")] # Absolute paths to actual PDFs
#     #     for pdf_file in pdf_files_to_link:
#     #         if pdf_file.exists():
#     #            (input_dir_path / pdf_file.name).symlink_to(pdf_file)
#     #         else:
#     #            print(f"Warning: Test PDF {pdf_file} not found.")
#     #
#     #     if any((input_dir_path / p.name).exists() for p in pdf_files_to_link):
#     #         batch_results, all_images = await parser.parse_pdfs_from_directory_input(pdf_files_to_link, input_dir_path)
#     #         for original_path, doc_data in batch_results:
#     #             if doc_data:
#     #                 print(f"Parsed {Path(original_path).name}: {doc_data.total_pages} pages")
#     #             else:
#     #                 print(f"Failed to parse {Path(original_path).name}")
#     #         print(f"Total unique image blobs from batch: {len(all_images)}")
#     #     else:
#     #         print("No valid PDFs symlinked for batch test.")
#
# if __name__ == "__main__":
#     # asyncio.run(main_test_single())
#     # asyncio.run(main_test_batch())
#     print("PDFParsingService defined. To test, call main_test_single() or main_test_batch() after setup.")

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
