import json
import logging
from typing import Any

from sda.core.models import Task
from sda.services.analysis import DuplicatePair, SemanticSearchResult
from sda.config import MAX_TOOL_OUTPUT_LENGTH


class AgentOutputFormatter:
    """
    Handles formatting of tool outputs for the LlamaIndex agent.
    """

    @staticmethod
    def format_output(result: Any) -> str:
        """
        Converts tool outputs into readable strings for the LLM.
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
