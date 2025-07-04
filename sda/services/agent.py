# sda/services/agent.py

import json
import logging
from typing import Any, List, TYPE_CHECKING, Callable, Generator

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from sda.core.models import Task
from sda.services.analysis import DuplicatePair, SemanticSearchResult
from sda.utils.limiter import RateLimiter

if TYPE_CHECKING:
    from sda.app import CodeAnalysisFramework

# A safe limit for the character length of a tool's output to avoid context window overflow.
MAX_TOOL_OUTPUT_LENGTH = 16000


class AgentManager:
    """
    Manages the lifecycle and execution of the LlamaIndex agent.
    This class abstracts the interaction with the LLM and its tool-using capabilities.
    """

    def __init__(self, framework: 'CodeAnalysisFramework', rate_limiter: RateLimiter):
        """
        Initializes the AgentManager.

        Args:
            framework: An instance of the main CodeAnalysisFramework to access its methods as tools.
            rate_limiter: The rate limiter to use for controlling LLM API calls.
        """
        self.framework = framework
        self.rate_limiter = rate_limiter
        logging.info("AgentManager initialized.")

    def _format_output(self, result: Any) -> str:
        """
        Helper to convert tool outputs into readable strings for the LLM.
        Includes a safeguard to truncate excessively long outputs.
        """
        if result is None:
            raw_output = "Operation completed, but no data was returned."
        elif isinstance(result, list) and not result:
            raw_output = "Operation completed successfully, but the result is an empty list."
        elif isinstance(result, Task):
            raw_output = json.dumps({
                "task_id": result.uuid, "name": result.name, "status": result.status,
                "message": "Task started. You can check its status later using the UI."
            }, indent=2)
        elif isinstance(result, list) and all(hasattr(item, 'to_dict') for item in result):
            raw_output = json.dumps([item.to_dict() for item in result], indent=2)
        elif isinstance(result, list) and all(isinstance(item, (DuplicatePair, SemanticSearchResult)) for item in result):
            raw_output = json.dumps([item.__dict__ for item in result], indent=2)
        elif isinstance(result, (list, dict)):
            try:
                raw_output = json.dumps(result, indent=2)
            except (TypeError, OverflowError):
                raw_output = str(result)  # Fallback for non-serializable objects
        elif hasattr(result, '__dict__'):
            raw_output = json.dumps(result.__dict__, indent=2, default=str)
        else:
            raw_output = str(result)

        # Safeguard: Truncate the output if it's excessively long.
        if len(raw_output) > MAX_TOOL_OUTPUT_LENGTH:
            logging.warning(f"Tool output was truncated from {len(raw_output)} to {MAX_TOOL_OUTPUT_LENGTH} characters.")
            return raw_output[:MAX_TOOL_OUTPUT_LENGTH] + "\n\n... (Output truncated due to excessive length)"

        return raw_output

    def _get_tools_for_context(self, repo_id: int, branch: str) -> List[FunctionTool]:
        """Creates and returns a list of tools with the current repo/branch context."""
        tools = [
            FunctionTool.from_defaults(fn=lambda q: self._format_output(self.framework.analysis_engine.search_chunks_by_symbol(repo_id, branch, q)),
                                       name="search_code_by_symbol",
                                       description="Finds code chunks semantically related to a symbol name (e.g., a function or class name)."),
            FunctionTool.from_defaults(
                fn=lambda snippet: self._format_output(self.framework.analysis_engine.find_similar_chunks_by_snippet(repo_id, branch, snippet)),
                name="find_similar_code_by_snippet", description="Finds code chunks semantically similar to a given code snippet."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.find_dead_code_for_repo(repo_id, branch, 'agent')), name="find_dead_code",
                                       description="Starts a task to find potentially unused functions and classes. Returns a task object."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.find_duplicate_code_for_repo(repo_id, branch, 'agent')),
                                       name="find_duplicate_code",
                                       description="Starts a task to find semantically similar code chunks. Returns a task object."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.get_cpg_analysis(repo_id, branch)), name="get_code_graph_analysis",
                                       description="Gets a textual analysis of the Code Property Graph, including central nodes."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.get_repository_stats(repo_id, branch)), name="get_repository_statistics",
                                       description="Retrieves statistics for the current repository and branch, like file count, lines of code, and language breakdown."),
            FunctionTool.from_defaults(fn=lambda symbol: self._format_output(self.framework.navigation_tools.find_symbol_definition(symbol, repo_id, branch)),
                                       name="find_symbol_definition", description="Locates the definition of a symbol (e.g., a function or class name)."),
            FunctionTool.from_defaults(fn=lambda symbol: self._format_output(self.framework.navigation_tools.find_symbol_references(symbol, repo_id, branch)),
                                       name="find_symbol_references", description="Finds all usages/references of a symbol."),
            FunctionTool.from_defaults(fn=lambda file_path: self._format_output(self.framework.navigation_tools.get_file_outline(file_path, repo_id, branch)),
                                       name="get_file_outline", description="Provides a hierarchical view of the symbols within a single file."),
            FunctionTool.from_defaults(fn=lambda symbol: self._format_output(self.framework.navigation_tools.get_call_hierarchy(symbol, repo_id, branch)),
                                       name="get_call_hierarchy",
                                       description="Traces incoming calls (callers) and outgoing calls (callees) for a specific function."),
            FunctionTool.from_defaults(
                fn=lambda file_path: self._format_output(self.framework.navigation_tools.analyze_dependencies(file_path, repo_id, branch)),
                name="analyze_file_dependencies", description="Analyzes a file's incoming and outgoing dependencies based on code calls."),
            FunctionTool.from_defaults(
                fn=lambda path_prefix=None: self._format_output(self.framework.list_files_in_repo(repo_id, branch, path_prefix)),
                name="list_files_in_repository",
                description="Lists files and subdirectories within a given 'path_prefix'. If 'path_prefix' is not provided, it lists the top-level contents of the repository. Directories are indicated by a trailing '/'. To explore the project structure, start with no prefix and then use the returned directory paths in subsequent calls."
            ),
            FunctionTool.from_defaults(fn=lambda file_path: self._format_output(self.framework.get_file_content_by_path(repo_id, branch, file_path)),
                                       name="get_file_content",
                                       description="Retrieves the full text content of a specific file. Always use 'list_files_in_repository' first to get the correct relative file path."),
            FunctionTool.from_defaults(fn=lambda queries: self._format_output(self.framework.search_files_for_text(repo_id, queries.split(','))),
                                       name="search_files_for_text",
                                       description="Performs a full-text search for one or more comma-separated keywords across all files in the repository. Useful for finding configuration strings, comments, or content in non-code files."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.get_repository_status(repo_id)), name="get_git_status",
                                       description="Gets the git status (modified, new, untracked files) for the current repository."),
            FunctionTool.from_defaults(fn=lambda: self._format_output(self.framework.get_repository_branches(repo_id)), name="list_repository_branches",
                                       description="Lists all available branches for the current repository."),
            FunctionTool.from_defaults(fn=self.framework.add_repository, name="add_repository",
                                       description="Clones a remote repo or registers a local one. Takes a URL or local path. Returns the newly added repository object."),
            FunctionTool.from_defaults(fn=lambda new_branch: self._format_output(self.framework.analyze_branch(repo_id, new_branch, 'agent')),
                                       name="analyze_branch",
                                       description="Starts a task to check out a specific branch and run the full incremental ingestion process. This is required to make the branch's contents searchable by other tools. Returns a task object.")
        ]
        return tools

    def run_chat_stream(self, repo_id: int, branch: str, query: str, chat_history: List[ChatMessage],
                        stream_callback: Callable[[str], None]) -> Generator[str, None, None]:
        """
        Runs a query through the agent, streaming back updates via the callback,
        and yielding the final response.
        """
        if not repo_id or not branch:
            yield "Error: Please select a repository and branch before using the agent."
            return

        tools = self._get_tools_for_context(repo_id, branch)

        system_prompt = ChatMessage(role="system", content="""You are an expert software development assistant.
IMPORTANT: Before using the tools `analyze_branch`, `find_dead_code`, or `find_duplicate_code`, you MUST ask the user for confirmation as these tasks can take a long time.
For all other tools, you can proceed directly.
When you are done, respond to the user with your final answer.""")

        final_chat_history = [system_prompt] + chat_history

        agent = ReActAgent.from_tools(tools=tools, llm=Settings.llm, chat_history=final_chat_history, verbose=True)

        try:
            # Use the built-in stream_chat method
            streaming_response = agent.stream_chat(query)
            
            # Stream the response
            current_response = ""
            for token in streaming_response.response_gen:
                current_response += token
                stream_callback(current_response)
                
            yield current_response
            
        except Exception as e:
            error_msg = f"Error in agent execution: {str(e)}"
            logging.error(error_msg, exc_info=True)
            stream_callback(error_msg)
            yield error_msg