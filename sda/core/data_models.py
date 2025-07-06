from typing import Optional, List, Dict, Any
from datetime import datetime # Added datetime import
from pydantic import BaseModel, Field

class DuplicatePair(BaseModel):
    """Represents a pair of semantically similar code chunks."""
    chunk_a_id: str
    chunk_b_id: str
    file_a: str
    file_b: str
    lines_a: str # "start-end"
    lines_b: str # "start-end"
    similarity: float
    content_a_snippet: Optional[str] = None
    content_b_snippet: Optional[str] = None
    chunk_a_token_count: Optional[int] = None
    chunk_b_token_count: Optional[int] = None
    # Consider adding chunk_a_line_count, chunk_b_line_count if different from lines_a/b range

# Pydantic models for API responses, especially for ORM conversion.

class TaskRead(BaseModel):
    id: int
    uuid: str
    repository_id: int
    parent_id: Optional[int] = None
    name: str
    created_by: str
    status: str
    progress: float
    message: Optional[str] = None
    # log_history: Optional[str] = None # Usually not sent to frontend unless specifically needed
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None # Keep as dict for flexibility
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None # Keep as dict

    children: List['TaskRead'] = [] # Recursive model

    class Config:
        from_attributes = True

# If using Pydantic v2, update_forward_refs is often not needed if types are correctly quoted.
# However, for older Pydantic or complex scenarios, it might be.
# TaskRead.model_rebuild() # For Pydantic v2, or TaskRead.update_forward_refs() for V1
# For Pydantic v2, this is typically handled automatically if type hints are correct.

class SemanticSearchResult(BaseModel):
    """Represents a single result from a semantic search."""
    file_path: str
    content: str
    start_line: Optional[int]
    end_line: Optional[int]
    score: float

class TransientNode(BaseModel):
    """A temporary, serializable representation of an AST node for pipeline processing."""
    node_id: str
    node_type: str
    name: Optional[str]
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    text_content: str
    signature: Optional[str]
    docstring: Optional[str]
    depth: int
    complexity_score: Optional[float]
    parent_id: Optional[str] = None
    repo_metadata: Dict[str, Any] = Field(default_factory=dict)

class TransientChunk(BaseModel):
    """A temporary, serializable representation of a code chunk for pipeline processing."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance_score: float = 0.5
    ast_node_ids: List[str] = Field(default_factory=list)
    parent_chunk_id: Optional[str] = None

class DirectoryNode(BaseModel):
    """Represents a directory in the repository's file tree."""
    path: str # Using str for path to simplify serialization, Path object can be reconstructed
    parent: Optional['DirectoryNode'] = None
    children: Dict[str, 'DirectoryNode'] = Field(default_factory=dict)
    file_count: int = 0
    total_size: int = 0

    @property
    def depth(self) -> int:
        # Assuming path is relative to repo root, or includes repo root
        # and we count parts from there.
        # If path is absolute, this logic might need adjustment
        # or Path(self.path).parts might be more robust.
        return self.path.count('/') + (1 if self.path and self.path != '.' else 0)

    def __repr__(self):
        return f"<DirectoryNode path={self.path} files={self.file_count} size={self.total_size}>"

DirectoryNode.model_rebuild()
