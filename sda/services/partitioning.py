# sda/services/partitioning.py

"""
Implements the smart partitioning logic to divide a repository into
logical, architecturally-significant database schemas.
"""
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

from sda.config import PartitioningConfig

@dataclass
class DirectoryNode:
    """Represents a directory in the repository's file tree."""
    path: Path
    parent: Optional['DirectoryNode'] = None
    children: Dict[str, 'DirectoryNode'] = field(default_factory=dict)
    file_count: int = 0
    total_size: int = 0

    @property
    def depth(self) -> int:
        return len(self.path.parts)

    def __repr__(self):
        return f"<DirectoryNode path={self.path} files={self.file_count} size={self.total_size}>"

class SmartPartitioningService:
    """
    Analyzes a repository's file structure to determine an optimal,
    load-balanced, and architecturally-aware schema partitioning strategy.
    """
    def __init__(self):
        logging.info("SmartPartitioningService initialized.")
        self.config = PartitioningConfig

    def _build_tree(self, file_paths: List[Tuple[Path, int]]) -> DirectoryNode:
        """Builds an in-memory tree representing the directory structure."""
        root = DirectoryNode(path=Path('.'))
        for file_path, file_size in file_paths:
            current = root
            for part in file_path.parent.parts:
                if part not in current.children:
                    new_path = current.path / part if str(current.path) != '.' else Path(part)
                    current.children[part] = DirectoryNode(path=new_path, parent=current)
                current = current.children[part]
            current.file_count += 1
            current.total_size += file_size
        return root

    def _aggregate_stats(self, node: DirectoryNode):
        """Recursively traverses the tree bottom-up to aggregate file counts and sizes."""
        for child in node.children.values():
            self._aggregate_stats(child)
            node.file_count += child.file_count
            node.total_size += child.total_size

    def _get_all_dirs(self, node: DirectoryNode) -> List[DirectoryNode]:
        """Flattens the tree into a list of all directory nodes."""
        dirs = [node]
        for child in node.children.values():
            dirs.extend(self._get_all_dirs(child))
        return dirs

    def _select_partitions(self, root: DirectoryNode) -> Set[Path]:
        """
        Selects the best non-overlapping directories to become partitions,
        using dynamically calculated thresholds for depth and file count.
        """
        all_dirs = self._get_all_dirs(root)
        if not all_dirs:
            return set()

        # --- Dynamic Threshold Calculation ---
        total_files = root.file_count
        avg_depth = sum(d.depth for d in all_dirs) / len(all_dirs) if all_dirs else 0

        dynamic_min_file_count = max(
            self.config.MIN_SCHEMA_FILE_COUNT_ABSOLUTE,
            total_files * self.config.MIN_SCHEMA_FILE_COUNT_RATIO
        )
        dynamic_max_depth = min(
            self.config.MAX_DEPTH_ABSOLUTE_CAP,
            math.ceil(avg_depth * self.config.MAX_DEPTH_AVG_MULTIPLIER)
        )
        
        logging.info(
            f"Dynamic partitioning thresholds: min_files={dynamic_min_file_count:.0f}, "
            f"max_depth={dynamic_max_depth} (avg_depth={avg_depth:.2f})"
        )
        # --- End Dynamic Calculation ---

        # Filter candidates based on dynamic thresholds
        candidates = [
            d for d in all_dirs
            if d.depth > 0
            and d.depth <= dynamic_max_depth
            and d.file_count >= dynamic_min_file_count
        ]

        target = self.config.TARGET_FILES_PER_SCHEMA

        def cost_function(d: DirectoryNode) -> float:
            """
            Calculates a cost for a directory, heavily penalizing sizes
            over the target and rewarding sizes close to the target from below.
            """
            count = d.file_count
            if count > target:
                # Penalize directories that are larger than the target.
                # The penalty increases the further over the target it is.
                return (count - target) * 1.5
            else:
                # Reward directories that are close to the target from below.
                # The closer to the target, the smaller the cost (better).
                return target - count

        # Sort candidates by the cost function (ascending), then by depth (shallower is better).
        # This prioritizes partitions that best fit the target size.
        candidates.sort(key=lambda d: (cost_function(d), d.depth))

        selected_partitions: Set[Path] = set()
        for candidate in candidates:
            # Check if this candidate is already a child of a selected partition
            is_sub_partition = any(candidate.path.is_relative_to(p) for p in selected_partitions)
            if not is_sub_partition:
                selected_partitions.add(candidate.path)
        
        return selected_partitions

    def generate_schema_map(self, repo_path_str: str, all_files: List[Path], repo_uuid: str) -> Dict[Path, str]:
        """
        The main public method. Generates a mapping from each file path to its
        assigned schema name.

        Returns:
            A dictionary mapping absolute Path objects to schema name strings.
        """
        repo_path = Path(repo_path_str)
        if not all_files:
            return {}
            
        file_stats = [(p.relative_to(repo_path), p.stat().st_size) for p in all_files]

        root_node = self._build_tree(file_stats)
        self._aggregate_stats(root_node)
        partition_paths = self._select_partitions(root_node)
        
        logging.info(f"Selected {len(partition_paths)} partitions: {[str(p) for p in sorted(list(partition_paths))]}")

        file_to_schema_map: Dict[Path, str] = {}
        
        # Helper to create a clean schema name from a path
        def to_schema_name(path: Path) -> str:
            path_str = str(path)
            if not path_str or path_str == '.':
                return f"repo_{repo_uuid[:8]}__root"
            
            clean_path = re.sub(r'[^a-z0-9_]+', '_', path_str.lower().replace('/', '_'))
            return f"repo_{repo_uuid[:8]}_{clean_path.strip('_')}"[:63]

        for file_abs_path in all_files:
            file_rel_path = file_abs_path.relative_to(repo_path)
            assigned_schema = None
            # Find the most specific partition this file belongs to
            for part_path in sorted(list(partition_paths), key=lambda p: len(p.parts), reverse=True):
                if file_rel_path.is_relative_to(part_path):
                    assigned_schema = to_schema_name(part_path)
                    break
            
            # If not in any partition, assign to the root schema
            if not assigned_schema:
                assigned_schema = to_schema_name(Path('.')) # _root schema

            file_to_schema_map[file_abs_path] = assigned_schema

        return file_to_schema_map