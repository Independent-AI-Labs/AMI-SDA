# Technical Debt and Architectural Issues

This document outlines known technical debt, architectural concerns, and areas for improvement within the SDA codebase. This is a living document and should be updated as new issues are identified or existing ones are resolved.

## 1. Database and Data Integrity

*   **Re-analysis Data Clearing**:
    *   While initial fixes for Dgraph (`RESOURCE_EXHAUSTED`) and Postgres (`UniqueViolation` for ASTNode, DBCodeChunk) have been implemented, the overall data lifecycle management during re-analysis needs a comprehensive review. This includes ensuring all relevant data (not just for specific tables) for a branch is cleanly and efficiently removed or updated.
    *   Consider a more robust strategy for Dgraph data removal, potentially involving Dgraph's bulk delete features more directly if batching UIDs still proves problematic for extremely large datasets.
*   **Task Data Integrity in History**:
    *   Some completed tasks in history show "Unknown" status or "Timing N/A". This points to potential issues in the task lifecycle management (`_start_task`, `_update_task`, `_complete_task` in `app.py`) where `status`, `completed_at`, or `started_at` fields might not be consistently set or persisted correctly for all task completion paths. This needs investigation to ensure all historical task records are accurate.
*   **Schema Management for Partitions**:
    *   The current partitioning strategy creates multiple schemas per repository. While this helps isolate data, it can complicate queries that need to span the entire repository (requiring iteration over schemas) and adds overhead to schema management itself. A review of the trade-offs and potential alternative data organization strategies (e.g., using table partitioning within fewer schemas) could be beneficial long-term.

## 2. Analysis Services (Dead Code, Duplicate Code)

*   **Accuracy of Dead/Duplicate Code Detection**:
    *   The current algorithms for dead code (nodes with no Dgraph incoming calls) and duplicate code (embedding-based similarity) yield a high number of false positives.
    *   **Dead Code**: The Dgraph-based approach is a good starting point but often misses context (e.g., dynamically called functions, entry points, library code not called within the repo but externally). It needs more sophisticated analysis, possibly integrating call graph analysis with an understanding of public APIs, entry points, and framework-specific conventions.
    *   **Duplicate Code**: Semantic similarity via embeddings can find near-duplicates but might also flag unrelated code with similar structure or vocabulary. Thresholds need tuning, and potentially combining with more traditional AST-based or token-based duplication detection could improve precision.
    *   Further work is needed to refine these algorithms or explore alternative strategies. The recent addition of debug information is only a first step.
*   **Configuration and Tunability**:
    *   Analysis services (especially dead/duplicate code) lack fine-grained configuration (e.g., sensitivity thresholds, ignore patterns specific to analysis types). Adding these would help users adapt the tools to their specific codebases.

## 3. UI and User Experience

*   **Control Panel vs. Task History Display**:
    *   The active task view in the Control Panel (dynamic, JS-driven) and the Task History view (static, API-driven Jinja templates) use different rendering mechanisms. While information consistency has been improved, the user request for them to use the "same component/template" implies a desire for a more unified look, feel, and potentially live update behavior for recent history items. This would be a significant UI refactor.
*   **Error Reporting and Guidance**:
    *   While some error messages have been improved (e.g., Embedding View for missing data), a more systematic approach to user-facing error reporting could be implemented, offering clearer explanations and actionable suggestions.
*   **Embedding View Implementation**:
    *   The "Embedding View" tab is currently a placeholder. The actual functionality to visualize AST node breakdowns and embeddings needs to be designed and implemented.

## 4. Ingestion Pipeline

*   **Idempotency and Efficiency of Re-ingestion**:
    *   The process of re-analyzing a branch should be fully idempotent and as efficient as possible. The recent fixes for unique key violations are a step in this direction. However, a full review of the ingestion pipeline (`sda/services/ingestion/`) might reveal further opportunities to optimize by avoiding redundant work or improving how changes are detected and processed.
*   **Large File Handling**:
    *   The system's behavior with extremely large individual files (e.g., auto-generated code, large data files mistakenly included in analysis) should be reviewed. This could impact parsing times, memory usage, and chunking effectiveness.
*   **Tree-sitter Grammar Specifics**:
    *   The `LANGUAGE_SETTINGS` in `sda/config.py` (defining chunkable nodes, identifier nodes, etc.) for each language are based on general assumptions. These may need refinement based on deeper analysis of each tree-sitter grammar's specific node types and how they best serve chunking and code understanding goals. This is particularly true for the newly added languages (Bash, Markdown, HTML, CSS).

## 5. Configuration Management

*   **Distributed Configuration**:
    *   Some configurations are spread (e.g., Dgraph client options in `db_management.py`, various batch sizes in `IngestionConfig`). Consolidating or providing a clearer override mechanism for system-level settings might improve maintainability.

## 6. Code Structure and Maintainability

*   **`sda/ui.py` Complexity**:
    *   `sda/ui.py` is a very large file handling many aspects of the UI. Breaking it down into smaller, more focused components or modules could improve readability and maintainability.
*   **Placeholder Features**:
    *   Functions like `AdvancedCodeNavigationTools.get_call_hierarchy` and `analyze_dependencies` are placeholders. These need full implementation.

This list is not exhaustive but captures the main points observed during recent development and feedback cycles.
