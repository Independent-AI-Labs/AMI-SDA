#!/usr/bin/env python3
"""
Advanced PDF Content Extractor & HTML Reconstructor
Uses geometric algorithms, spatial analysis, and clustering for superior text ordering.
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed or conflicting package detected.")
    print("Please install with: pip install PyMuPDF")
    print("If you have an old 'fitz' package, uninstall it first: pip uninstall fitz")
    sys.exit(1)

import os
import sys
import json
import base64
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Optional dependencies for advanced clustering
try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class TextBlock:
    """Represents a text block with positioning and formatting information."""
    
    def __init__(self, text: str, bbox: Tuple[float, float, float, float], 
                 font_info: Dict, page_num: int):
        self.text = text.strip()
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.font_info = font_info
        self.page_num = page_num
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.center_x = (self.x0 + self.x1) / 2
        self.center_y = (self.y0 + self.y1) / 2
        
    def __repr__(self):
        return f"TextBlock('{self.text[:30]}...', bbox={self.bbox})"

class ColumnDetector:
    """Detects column structures using improved geometric analysis."""
    
    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height
        self.min_column_width = page_width * 0.15  # Minimum 15% of page width
        
    def detect_columns(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Detect columns using enhanced vertical projection analysis."""
        if not text_blocks:
            return []
        
        # Try simple column detection first
        columns = self._detect_simple_columns(text_blocks)
        
        # If we get too many tiny columns, fall back to single column
        if len(columns) > 3 or any(len(col) < 2 for col in columns):
            return [text_blocks]
        
        return columns if columns else [text_blocks]
    
    def _detect_simple_columns(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Simple column detection using text block distribution."""
        if not text_blocks:
            return []
        
        # Group blocks by approximate horizontal position
        x_positions = [block.center_x for block in text_blocks]
        
        # Find natural breaks in x positions
        sorted_x = sorted(x_positions)
        gaps = []
        
        for i in range(1, len(sorted_x)):
            gap = sorted_x[i] - sorted_x[i-1]
            if gap > 50:  # Significant gap
                gaps.append((sorted_x[i-1] + gap/2, gap))
        
        # If no significant gaps, return single column
        if not gaps:
            return [text_blocks]
        
        # Use the largest gap as column separator
        largest_gap = max(gaps, key=lambda g: g[1])
        separator_x = largest_gap[0]
        
        # Split blocks into columns
        left_column = [block for block in text_blocks if block.center_x < separator_x]
        right_column = [block for block in text_blocks if block.center_x >= separator_x]
        
        columns = []
        if left_column:
            columns.append(left_column)
        if right_column:
            columns.append(right_column)
        
        return columns
    
    def _create_vertical_profile(self, text_blocks: List[TextBlock]) -> np.ndarray:
        """Create vertical projection profile showing text density."""
        profile = np.zeros(int(self.page_width))
        
        for block in text_blocks:
            start_x = max(0, int(block.x0))
            end_x = min(int(self.page_width), int(block.x1))
            if start_x < end_x:
                profile[start_x:end_x] += 1
            
        return profile
    
    def _find_column_boundaries(self, profile: np.ndarray) -> List[float]:
        """Find column boundaries by detecting valleys in the profile."""
        if len(profile) == 0:
            return []
        
        # Smooth the profile
        kernel_size = max(5, int(self.page_width * 0.01))
        if kernel_size < len(profile):
            smoothed = np.convolve(profile, np.ones(kernel_size)/kernel_size, mode='same')
        else:
            smoothed = profile
        
        # Find valleys (local minima)
        valleys = []
        min_valley_width = int(self.page_width * 0.05)  # 5% of page width
        
        for i in range(min_valley_width, len(smoothed) - min_valley_width):
            if smoothed[i] == 0:
                # Check if this is a significant valley
                valley_width = 1
                for j in range(1, min_valley_width):
                    if i-j >= 0 and smoothed[i-j] == 0:
                        valley_width += 1
                    else:
                        break
                
                if valley_width >= min_valley_width:
                    valleys.append(i)
        
        # Remove adjacent valleys
        filtered_valleys = []
        for valley in valleys:
            if not filtered_valleys or valley - filtered_valleys[-1] > min_valley_width:
                filtered_valleys.append(valley)
        
        return sorted(filtered_valleys)
    
    def _group_blocks_by_columns(self, text_blocks: List[TextBlock], 
                                boundaries: List[float]) -> List[List[TextBlock]]:
        """Group text blocks into columns based on boundaries."""
        if not boundaries:
            return [text_blocks]
        
        columns = [[] for _ in range(len(boundaries) + 1)]
        
        for block in text_blocks:
            column_idx = 0
            for i, boundary in enumerate(boundaries):
                if block.center_x > boundary:
                    column_idx = i + 1
                else:
                    break
            columns[column_idx].append(block)
        
        return [col for col in columns if col]  # Remove empty columns

class ReadingOrderDetector:
    """Determines proper reading order using improved geometric algorithms."""
    
    def __init__(self):
        self.line_tolerance = 5  # pixels - stricter line detection
        self.paragraph_gap = 15  # pixels - gap indicating new paragraph
        
    def sort_blocks_reading_order(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort text blocks in proper reading order with improved line detection."""
        if not text_blocks:
            return []
        
        # Convert PDF coordinates (bottom-left origin) to top-left for easier processing
        max_y = max(block.y1 for block in text_blocks)
        for block in text_blocks:
            block.normalized_y0 = max_y - block.y1  # Flip Y coordinate
            block.normalized_y1 = max_y - block.y0
        
        # Sort by normalized y position (top to bottom)
        sorted_blocks = sorted(text_blocks, key=lambda b: b.normalized_y0)
        
        # Group blocks into lines using improved algorithm
        lines = self._group_into_lines_improved(sorted_blocks)
        
        # Sort each line left to right and combine
        ordered_blocks = []
        for line in lines:
            # Sort line blocks by x position
            line.sort(key=lambda b: b.x0)
            ordered_blocks.extend(line)
        
        return ordered_blocks
    
    def _group_into_lines_improved(self, blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Improved line grouping using text baseline analysis."""
        if not blocks:
            return []
        
        lines = []
        current_line = [blocks[0]]
        
        for block in blocks[1:]:
            # Check if block belongs to current line
            if self._belongs_to_same_line(current_line, block):
                current_line.append(block)
            else:
                # Check if this is a new paragraph (large gap)
                if self._is_new_paragraph(current_line, block):
                    lines.append(current_line)
                    current_line = [block]
                else:
                    lines.append(current_line)
                    current_line = [block]
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _belongs_to_same_line(self, current_line: List[TextBlock], block: TextBlock) -> bool:
        """Check if block belongs to the same line as current line blocks."""
        if not current_line:
            return False
        
        # Use the most recent block in the line for comparison
        ref_block = current_line[-1]
        
        # Calculate vertical distance between text baselines
        ref_baseline = ref_block.normalized_y1  # Bottom of reference block
        block_baseline = block.normalized_y1    # Bottom of new block
        
        # Check if baselines align within tolerance
        baseline_diff = abs(ref_baseline - block_baseline)
        
        # Also check if there's significant overlap in y-range
        ref_height = ref_block.normalized_y1 - ref_block.normalized_y0
        block_height = block.normalized_y1 - block.normalized_y0
        avg_height = (ref_height + block_height) / 2
        
        # Consider same line if baseline difference is small relative to text height
        return baseline_diff <= max(self.line_tolerance, avg_height * 0.3)
    
    def _is_new_paragraph(self, current_line: List[TextBlock], block: TextBlock) -> bool:
        """Check if there's a significant gap indicating a new paragraph."""
        if not current_line:
            return False
        
        # Calculate gap between last line and new block
        last_line_bottom = max(b.normalized_y1 for b in current_line)
        gap = block.normalized_y0 - last_line_bottom
        
        return gap > self.paragraph_gap

class AdvancedTextExtractor:
    """Advanced text extraction with improved spatial analysis."""
    
    def __init__(self):
        self.column_detector = None
        self.reading_order_detector = ReadingOrderDetector()
        
    def extract_text_blocks(self, page: fitz.Page) -> List[TextBlock]:
        """Extract text blocks using multiple PyMuPDF methods for better accuracy."""
        
        # Method 1: Use get_text("dict") for detailed structure
        text_dict = page.get_text("dict")
        dict_blocks = self._extract_from_dict(text_dict, page.number)
        
        # Method 2: Use get_text("words") for word-level precision
        words = page.get_text("words")
        word_blocks = self._extract_from_words(words, page.number)
        
        # Combine and deduplicate
        all_blocks = dict_blocks + word_blocks
        
        # Remove duplicates and very short blocks
        filtered_blocks = self._filter_and_deduplicate(all_blocks)
        
        # Merge nearby blocks that should be together
        merged_blocks = self._merge_nearby_blocks(filtered_blocks)
        
        return merged_blocks
    
    def _extract_from_dict(self, text_dict: dict, page_num: int) -> List[TextBlock]:
        """Extract from hierarchical dict structure."""
        blocks = []
        
        for block_num, block in enumerate(text_dict["blocks"]):
            if block["type"] == 0:  # Text block
                block_text = ""
                font_info = {"fonts": [], "sizes": [], "colors": [], "flags": []}
                
                # Extract text and collect font information
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_info["fonts"].append(span["font"])
                        font_info["sizes"].append(span["size"])
                        font_info["colors"].append(span["color"])
                        font_info["flags"].append(span["flags"])
                    block_text += line_text + " "
                
                if block_text.strip():
                    dominant_font = self._get_dominant_font_info(font_info)
                    text_block = TextBlock(
                        text=block_text.strip(),
                        bbox=block["bbox"],
                        font_info=dominant_font,
                        page_num=page_num
                    )
                    blocks.append(text_block)
        
        return blocks
    
    def _extract_from_words(self, words: list, page_num: int) -> List[TextBlock]:
        """Extract word-level blocks for fine-grained control."""
        if not words:
            return []
        
        # Group words into logical text blocks
        word_blocks = []
        current_group = []
        
        for word in words:
            if len(word) >= 7:  # word format: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
                if current_group and self._should_start_new_group(current_group[-1], word):
                    # Process current group
                    if current_group:
                        word_blocks.append(self._create_block_from_words(current_group, page_num))
                    current_group = [word]
                else:
                    current_group.append(word)
        
        # Process final group
        if current_group:
            word_blocks.append(self._create_block_from_words(current_group, page_num))
        
        return word_blocks
    
    def _should_start_new_group(self, prev_word: tuple, curr_word: tuple) -> bool:
        """Determine if we should start a new text block."""
        prev_x0, prev_y0, prev_x1, prev_y1 = prev_word[:4]
        curr_x0, curr_y0, curr_x1, curr_y1 = curr_word[:4]
        
        # Check for significant horizontal or vertical gaps
        horizontal_gap = curr_x0 - prev_x1
        vertical_gap = abs(curr_y0 - prev_y0)
        
        # Start new group if there's a large gap
        return horizontal_gap > 20 or vertical_gap > 5
    
    def _create_block_from_words(self, words: list, page_num: int) -> TextBlock:
        """Create a TextBlock from a group of words."""
        text_parts = []
        min_x0 = float('inf')
        min_y0 = float('inf')
        max_x1 = float('-inf')
        max_y1 = float('-inf')
        
        for word in words:
            x0, y0, x1, y1, text = word[:5]
            text_parts.append(text)
            min_x0 = min(min_x0, x0)
            min_y0 = min(min_y0, y0)
            max_x1 = max(max_x1, x1)
            max_y1 = max(max_y1, y1)
        
        combined_text = " ".join(text_parts)
        bbox = (min_x0, min_y0, max_x1, max_y1)
        
        # Simple font info for word-based blocks
        font_info = {"font": "Arial", "size": 12, "color": 0, "flags": 0}
        
        return TextBlock(
            text=combined_text,
            bbox=bbox,
            font_info=font_info,
            page_num=page_num
        )
    
    def _filter_and_deduplicate(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Remove duplicates and very short blocks."""
        filtered = []
        
        for block in blocks:
            # Skip very short blocks
            if len(block.text.strip()) < 2:
                continue
            
            # Check for duplicates
            is_duplicate = False
            for existing in filtered:
                if (abs(block.x0 - existing.x0) < 2 and 
                    abs(block.y0 - existing.y0) < 2 and
                    block.text.strip() == existing.text.strip()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(block)
        
        return filtered
    
    def _merge_nearby_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Merge blocks that are very close and should be together."""
        if not blocks:
            return []
        
        merged = []
        remaining = blocks[:]
        
        while remaining:
            current = remaining.pop(0)
            
            # Find blocks to merge with current
            to_merge = [current]
            i = 0
            while i < len(remaining):
                if self._should_merge_blocks(current, remaining[i]):
                    to_merge.append(remaining.pop(i))
                else:
                    i += 1
            
            # Merge blocks if we found any
            if len(to_merge) > 1:
                merged_block = self._merge_blocks(to_merge)
                merged.append(merged_block)
            else:
                merged.append(current)
        
        return merged
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Check if two blocks should be merged."""
        # Check if blocks are on the same line and close together
        vertical_overlap = min(block1.y1, block2.y1) - max(block1.y0, block2.y0)
        horizontal_gap = abs(block1.x1 - block2.x0) if block1.x1 < block2.x0 else abs(block2.x1 - block1.x0)
        
        return vertical_overlap > 0 and horizontal_gap < 10
    
    def _merge_blocks(self, blocks: List[TextBlock]) -> TextBlock:
        """Merge multiple blocks into one."""
        # Sort blocks by x position
        blocks.sort(key=lambda b: b.x0)
        
        # Combine text
        combined_text = " ".join(block.text for block in blocks)
        
        # Calculate combined bbox
        min_x0 = min(block.x0 for block in blocks)
        min_y0 = min(block.y0 for block in blocks)
        max_x1 = max(block.x1 for block in blocks)
        max_y1 = max(block.y1 for block in blocks)
        
        bbox = (min_x0, min_y0, max_x1, max_y1)
        
        return TextBlock(
            text=combined_text,
            bbox=bbox,
            font_info=blocks[0].font_info,  # Use first block's font info
            page_num=blocks[0].page_num
        )
    
    def _get_dominant_font_info(self, font_info: Dict) -> Dict:
        """Extract dominant font characteristics from collected font info."""
        if not font_info["fonts"]:
            return {"font": "Arial", "size": 12, "color": 0, "flags": 0}
        
        # Find most common font
        from collections import Counter
        font_counter = Counter(font_info["fonts"])
        dominant_font = font_counter.most_common(1)[0][0]
        
        # Calculate average size
        avg_size = sum(font_info["sizes"]) / len(font_info["sizes"])
        
        # Most common color
        color_counter = Counter(font_info["colors"])
        dominant_color = color_counter.most_common(1)[0][0]
        
        # Most common flags
        flags_counter = Counter(font_info["flags"])
        dominant_flags = flags_counter.most_common(1)[0][0]
        
        return {
            "font": dominant_font,
            "size": avg_size,
            "color": dominant_color,
            "flags": dominant_flags
        }

class SpatialTextClustering:
    """Clusters text blocks using spatial relationships."""
    
    def __init__(self):
        self.eps = 20  # DBSCAN epsilon parameter
        self.min_samples = 2  # DBSCAN minimum samples
        
    def cluster_related_blocks(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Cluster spatially related text blocks."""
        if not text_blocks or not HAS_SKLEARN:
            return [text_blocks]
        
        # Extract coordinates (using center points)
        coordinates = np.array([[block.center_x, block.center_y] for block in text_blocks])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coordinates)
        
        # Group blocks by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            clusters[label].append(text_blocks[i])
        
        # Convert to list, handling noise points (label -1)
        cluster_list = []
        for label, blocks in clusters.items():
            if label != -1:  # Not noise
                cluster_list.append(blocks)
            else:  # Noise points - add individually
                cluster_list.extend([[block] for block in blocks])
        
        return cluster_list

def extract_content_from_pdf(pdf_path: str, output_dir: str = None) -> str:
    """Extract content with advanced spatial analysis and text ordering."""
    
    # Set default output directory
    if output_dir is None:
        pdf_name = Path(pdf_path).stem
        output_dir = f"{pdf_name}_advanced_extracted"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return ""
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize processors
    text_extractor = AdvancedTextExtractor()
    spatial_clustering = SpatialTextClustering()
    
    all_content = []
    image_count = 0
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_rect = page.rect
        
        print(f"\nProcessing page {page_num + 1}...")
        
        # Initialize column detector for this page
        column_detector = ColumnDetector(page_rect.width, page_rect.height)
        
        # Extract images (same as before)
        page_images = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            filename = f"page_{page_num + 1:03d}_img_{img_index + 1:03d}.{image_ext}"
            filepath = os.path.join(images_dir, filename)
            
            with open(filepath, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Get image positions
            image_rects = page.get_image_rects(xref)
            for rect in image_rects:
                page_images.append({
                    "type": "image",
                    "filename": filename,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "width": rect.width,
                    "height": rect.height,
                    "y_position": rect.y0,
                    "image_data": base64.b64encode(image_bytes).decode('utf-8'),
                    "image_ext": image_ext
                })
            
            image_count += 1
        
        # Extract text blocks using improved methods
        text_blocks = text_extractor.extract_text_blocks(page)
        print(f"  Extracted {len(text_blocks)} text blocks")
        
        # Debug: Show some sample blocks
        for i, block in enumerate(text_blocks[:3]):
            print(f"    Block {i}: '{block.text[:50]}...' at ({block.x0:.1f}, {block.y0:.1f})")
        
        # Detect columns
        columns = column_detector.detect_columns(text_blocks)
        print(f"  Detected {len(columns)} column(s)")
        
        # Debug: Show column information
        for i, column in enumerate(columns):
            print(f"    Column {i}: {len(column)} blocks, x-range: {min(b.x0 for b in column):.1f}-{max(b.x1 for b in column):.1f}")
        
        # Process each column separately
        ordered_elements = []
        for col_idx, column_blocks in enumerate(columns):
            if not column_blocks:
                continue
            
            print(f"  Processing column {col_idx}...")
            
            # Apply reading order detection within column
            ordered_column = text_extractor.reading_order_detector.sort_blocks_reading_order(column_blocks)
            
            # Debug: Show first few blocks in reading order
            print(f"    First 3 blocks in reading order:")
            for i, block in enumerate(ordered_column[:3]):
                print(f"      {i}: '{block.text[:30]}...' at ({block.x0:.1f}, {block.y0:.1f})")
            
            ordered_elements.extend(ordered_column)
        
        # Convert TextBlocks to dictionary format
        text_elements = []
        for block in ordered_elements:
            text_elements.append({
                "type": "text",
                "text": block.text,
                "bbox": block.bbox,
                "font_info": block.font_info,
                "y_position": block.y0,
                "x_position": block.x0
            })
        
        # Combine text and images, sort by position
        all_elements = text_elements + page_images
        all_elements.sort(key=lambda x: (-x["y_position"], x.get("x_position", x["bbox"][0])))
        
        # Debug: Show final ordering
        print(f"  Final element order (first 5):")
        for i, elem in enumerate(all_elements[:5]):
            if elem["type"] == "text":
                print(f"    {i}: TEXT: '{elem['text'][:40]}...' at ({elem['x_position']:.1f}, {elem['y_position']:.1f})")
            else:
                print(f"    {i}: IMAGE: {elem['filename']} at ({elem['bbox'][0]:.1f}, {elem['bbox'][1]:.1f})")
        
        all_content.append({
            "page": page_num + 1,
            "elements": all_elements,
            "columns_detected": len(columns),
            "page_dimensions": {"width": page_rect.width, "height": page_rect.height}
        })
        
        print(f"  Page {page_num + 1} complete: {len(all_elements)} elements total")
    
    # Generate enhanced HTML
    html_content = generate_enhanced_html(all_content, pdf_path)
    
    # Save files
    html_file = os.path.join(output_dir, "reconstructed_advanced.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Save detailed metadata
    metadata_file = os.path.join(output_dir, "advanced_metadata.json")
    metadata = {
        "source_pdf": pdf_path,
        "processing_method": "advanced_spatial_analysis",
        "total_pages": len(doc),
        "total_images": image_count,
        "features_used": [
            "column_detection",
            "spatial_clustering", 
            "reading_order_detection",
            "advanced_text_extraction"
        ],
        "content": all_content
    }
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    doc.close()
    
    print(f"\nAdvanced extraction complete!")
    print(f"Total images extracted: {image_count}")
    print(f"HTML reconstruction: {html_file}")
    print(f"Metadata saved: {metadata_file}")
    
    return output_dir

def generate_enhanced_html(all_content: List[Dict], pdf_path: str) -> str:
    """Generate enhanced HTML with improved layout and formatting."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Reconstruction: {Path(pdf_path).stem}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        .page {{
            border: 1px solid #ddd;
            margin: 30px 0;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .page-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            margin: -30px -30px 30px -30px;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .page-stats {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .text-block {{
            margin: 15px 0;
            padding: 10px;
            border-left: 3px solid #f0f0f0;
            transition: all 0.3s ease;
        }}
        .text-block:hover {{
            background: #f8f9fa;
            border-left-color: #667eea;
        }}
        .image-block {{
            margin: 20px 0;
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .image-block img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .processing-info {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196F3;
        }}
        .processing-info h3 {{
            margin-top: 0;
            color: #1976D2;
        }}
        .features-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .feature-tag {{
            background: #2196F3;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .font-preserved {{
            /* Preserve original font characteristics */
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            .page {{
                padding: 20px;
            }}
            .page-header {{
                margin: -20px -20px 20px -20px;
                padding: 10px 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="processing-info">
        <h3>🚀 Advanced PDF Reconstruction</h3>
        <p><strong>Source:</strong> {pdf_path}</p>
        <p><strong>Processing Method:</strong> Advanced Spatial Analysis with Geometric Algorithms</p>
        <div class="features-list">
            <span class="feature-tag">Column Detection</span>
            <span class="feature-tag">Spatial Clustering</span>
            <span class="feature-tag">Reading Order Detection</span>
            <span class="feature-tag">Advanced Text Extraction</span>
            <span class="feature-tag">Font Preservation</span>
        </div>
    </div>
"""
    
    for page_content in all_content:
        html += f"""
    <div class="page">
        <div class="page-header">
            <span>Page {page_content['page']}</span>
            <div class="page-stats">
                {page_content['columns_detected']} columns • 
                {len(page_content['elements'])} elements • 
                {page_content['page_dimensions']['width']:.0f}×{page_content['page_dimensions']['height']:.0f}px
            </div>
        </div>
"""
        
        for element in page_content["elements"]:
            if element["type"] == "text":
                # Apply original font styling
                font_info = element.get("font_info", {})
                font_family = sanitize_font_family(font_info.get("font", "Arial"))
                font_size = font_info.get("size", 12)
                font_weight = get_font_weight(font_info.get("flags", 0))
                font_style = get_font_style(font_info.get("flags", 0))
                color = format_color(font_info.get("color", 0))
                
                style = f"font-family: {font_family}; font-size: {font_size}pt; font-weight: {font_weight}; font-style: {font_style}; color: {color};"
                
                text = element["text"].replace("\n", "<br>").strip()
                if text:
                    html += f"""        <div class="text-block">
            <div class="font-preserved" style="{style}">{text}</div>
        </div>
"""
            
            elif element["type"] == "image":
                html += f"""        <div class="image-block">
            <img src="data:image/{element['image_ext']};base64,{element['image_data']}" 
                 alt="Image from page {page_content['page']}" 
                 style="max-width: {element['width']}px;">
        </div>
"""
        
        html += "    </div>\n"
    
    html += """
    <div class="processing-info">
        <h3>📊 Processing Summary</h3>
        <p>This document was reconstructed using advanced geometric algorithms including:</p>
        <ul>
            <li><strong>Column Detection:</strong> Vertical projection profile analysis</li>
            <li><strong>Spatial Clustering:</strong> DBSCAN-based text block grouping</li>
            <li><strong>Reading Order:</strong> Geometric line detection and left-to-right sorting</li>
            <li><strong>Font Preservation:</strong> Original typeface, size, and color extraction</li>
        </ul>
    </div>
</body>
</html>"""
    
    return html

def sanitize_font_family(font_name: str) -> str:
    """Clean up font name for CSS."""
    if not font_name:
        return "Arial, sans-serif"
    
    # Remove common PDF font prefixes
    font_name = font_name.split('+')[-1]
    
    # Map common PDF fonts to web-safe fonts
    font_mapping = {
        "Times": "Times New Roman, serif",
        "TimesNewRoman": "Times New Roman, serif", 
        "Helvetica": "Arial, sans-serif",
        "Arial": "Arial, sans-serif",
        "Courier": "Courier New, monospace",
        "Symbol": "Symbol, sans-serif"
    }
    
    for pdf_font, css_font in font_mapping.items():
        if pdf_font.lower() in font_name.lower():
            return css_font
    
    # Fallback: use the font name with a generic fallback
    if "serif" in font_name.lower():
        return f"'{font_name}', serif"
    elif "mono" in font_name.lower() or "courier" in font_name.lower():
        return f"'{font_name}', monospace"
    else:
        return f"'{font_name}', sans-serif"

def get_font_weight(flags: int) -> str:
    """Convert PDF font flags to CSS font-weight."""
    if flags & 2**4:  # Bold flag
        return "bold"
    return "normal"

def get_font_style(flags: int) -> str:
    """Convert PDF font flags to CSS font-style."""
    if flags & 2**6:  # Italic flag
        return "italic"
    return "normal"

def format_color(color: int) -> str:
    """Convert PDF color to CSS hex color."""
    if isinstance(color, int):
        return f"#{color:06x}" if color > 0 else "#000000"
    return "#000000"

def main():
    # Set default arguments if not provided
    if len(sys.argv) < 2:
        sys.argv.append(r"C:\Users\vdonc\mco-ai-poc\data\knowledge_base\[KB-1] MCO Knowledge Base (cut-off 29-04-2025).pdf")
        sys.argv.append(r"C:\Users\vdonc\mco-ai-poc\data\knowledge_base\images")
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_pdf_extractor.py <pdf_file> [output_directory]")
        print("Example: python advanced_pdf_extractor.py document.pdf")
        print("Example: python advanced_pdf_extractor.py document.pdf extracted_content")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    if not HAS_SKLEARN:
        print("Note: sklearn not available. Advanced clustering features will be limited.")
        print("Install with: pip install scikit-learn")
    
    extract_content_from_pdf(pdf_path, output_dir)

if __name__ == "__main__":
    main()
