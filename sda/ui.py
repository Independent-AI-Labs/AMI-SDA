# ui.py

"""
Implements the Gradio-based user interface for the Code Analysis Framework.

This dashboard provides tools for repository management, an overview of
analysis insights, a live code editor, and version control operations.
It interacts exclusively with the CodeAnalysisFramework facade.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Generator, Optional
import os
import asyncio # Added for asyncio.create_task
import psutil # For CPU load and RAM
import re # For _extract_progress_from_html
from datetime import datetime, timezone # For time elapsed
import logging # Added import for logging
try:
    import torch
except ImportError:
    torch = None # type: ignore

import gradio as gr
import pandas as pd
import plotly.express as px
from PIL import Image
from llama_index.core.llms import ChatMessage
from jinja2 import Environment, FileSystemLoader, select_autoescape

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import Query, HTTPException # Added HTTPException
from sqlalchemy.orm import joinedload # Added for eager loading

from app import CodeAnalysisFramework

# Task Name Mapping
TASK_NAME_MAPPING = {
    "find_dead_code": "Find Unused Code",
    "find_duplicate_code": "Find Duplicate Code",
    "analyze_branch": "Analyze Branch Content", # More specific for the button
    # If ingestion_service.py creates tasks like "partition_repository", "parse_and_chunk_files", etc.
    # they can be mapped here if they become parent tasks visible in UI.
    # For now, focusing on user-initiated tasks and top-level ingestion.
}

def get_display_task_name(task_name_internal: Optional[str]) -> str:
    if not task_name_internal:
        return "Unknown Task"
    if task_name_internal.startswith("Ingest Branch:"):
        branch = task_name_internal.replace("Ingest Branch:", "").strip()
        return f"Ingesting Branch: {branch}" # Or "Analyzing Branch: {branch}"
    return TASK_NAME_MAPPING.get(task_name_internal, task_name_internal)

from sda.core.models import Task as SQLA_Task, Task  # Alias to avoid confusion if TaskRead is also named Task
from sda.core.data_models import TaskRead # Import the new Pydantic model
from sda.config import IngestionConfig, AIConfig, PG_DB_NAME, DGRAPH_HOST, DGRAPH_PORT
from sda.utils.websocket_manager import control_panel_manager


from sda.utils.websocket_manager import control_panel_manager


# WebSocket endpoint for the Control Panel
main_event_loop = None # Global variable to store the main event loop

async def websocket_control_panel_endpoint(websocket: WebSocket):
    await control_panel_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive, waiting for messages or disconnect
            # We don't expect messages from client here, but FastAPI requires a receive loop
            # or it will close the connection.
            data = await websocket.receive_text()
            # Optionally, log or handle client messages if any are expected in the future
            # print(f"Received message from Control Panel client: {data}")
    except WebSocketDisconnect:
        control_panel_manager.disconnect(websocket)
        # print(f"Control Panel client disconnected: {websocket.client}") # Disabled verbose log
    except Exception as e:
        print(f"Error in Control Panel WebSocket: {e}") # Keep actual error logging
        control_panel_manager.disconnect(websocket)


class DashboardUI:
    """Manages the Gradio UI components and their interactions."""

    def __init__(self, framework: CodeAnalysisFramework):
        self.framework = framework
        # It's important that the framework instance is accessible for polling logic
        # that will eventually call control_panel_manager.broadcast()
        # This might require passing `self` or `framework` to the polling function
        # or making `control_panel_manager` accessible to it.
        # For now, `handle_polling` is a method of `DashboardUI`, so it has `self.framework`.
        self.task_buttons: List[gr.Button] = []
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def _get_task_button_updates(self, interactive: bool) -> Tuple[gr.update, ...]:
        """Helper to generate a tuple of updates for all task-related buttons."""
        return tuple(gr.update(interactive=interactive) for _ in self.task_buttons)

    def _extract_progress_from_html(self, html_content: str) -> float:
        """Extract progress percentage from HTML content for comparison."""
        try:
            # import re # Moved to top-level imports
            match = re.search(r'data-content-hash="([^"]*)"', html_content)
            if match:
                content_hash = match.group(1)
                progress_str = content_hash.split('_')[0]
                return float(progress_str)
        except:
            pass
        return 0.0

    def _create_html_progress_bar(self, progress: float, message: str = "", task_name: str = "", unique_prefix: str = "generic") -> str:
        """Creates an HTML progress bar with CSS styling and JavaScript animations."""
        progress = max(0, min(100, progress))
        
        # The content_hash logic for diffing might be less relevant now,
        # or could be handled differently if Gradio's HTML component updates efficiently.
        # For now, we remove it from the template itself.
        template = self.jinja_env.get_template("progress_bar.html")
        return template.render(progress=progress, message=message, task_name=task_name, unique_prefix=unique_prefix)

    def _create_status_progress_html(self, task) -> str:
        """Creates detailed HTML progress display for the status modal."""
        if not task:
            return f"""<div class='status-container'>
No active tasks.
<div class='model-info-container'>
    <h4>Model Information</h4>
    <p><strong>LLM:</strong> {AIConfig.ACTIVE_LLM_MODEL}</p>
    <p><strong>Embedding Model:</strong> {AIConfig.ACTIVE_EMBEDDING_MODEL}</p>
    <p><strong>Embedding Devices:</strong> {AIConfig.EMBEDDING_DEVICES}</p>
</div>
{self._get_hardware_info_html()}
{self._get_storage_info_html()}
{self._get_usage_stats_html()}
</div>""" # This was the end of the 'if not task:' block

        # Prepare context for the main template
        context = {
            "task": task, # Pass the original task object for other attributes
            "display_task_name": get_display_task_name(task.name), # Use display name for title
            "model_info_html": self._get_model_info_html(),
            "hardware_info_html": self._get_hardware_info_html(),
            "storage_info_html": self._get_storage_info_html(),
            "usage_stats_html": self._get_usage_stats_html(),
            "main_task_progress_html": self._create_html_progress_bar(task.progress, task.message, get_display_task_name(task.name)),
            "task_timing_html": self._get_task_timing_html(task),
            "render_sub_task": self._render_sub_task_html # Pass method to render sub_tasks
        }
        
        template = self.jinja_env.get_template("status_container_base.html")
        return template.render(context)

    def _render_sub_task_html(self, child_task: SQLA_Task) -> str: # Changed Task to SQLA_Task
        """Renders a single sub-task using its template."""
        display_name = get_display_task_name(child_task.name)
        progress_bar_html = self._create_html_progress_bar(child_task.progress, child_task.message, display_name)
        template = self.jinja_env.get_template("status_modal_parts/sub_task.html")
        # Pass both original task and display_name to template context if needed,
        # or just ensure template uses the display_name for rendering the name.
        # Assuming template uses task.name, we might need to adjust template or pass display_name explicitly.
        # For now, the progress_bar_html receives the display_name.
        # If the template `sub_task.html` also renders `task.name` directly, it needs adjustment or `display_name` in its context.
        # Let's assume the template is simple and primarily shows the progress bar's title.
        # To be safe, let's prepare for the template to use a display_name variable.
        return template.render(task=child_task, display_task_name=display_name, progress_bar_html=progress_bar_html)

    def _get_model_info_html(self) -> str:
        template = self.jinja_env.get_template("status_modal_parts/model_info.html")
        return template.render(
            active_llm_model=AIConfig.ACTIVE_LLM_MODEL,
            active_embedding_model=AIConfig.ACTIVE_EMBEDDING_MODEL,
            embedding_devices=AIConfig.EMBEDDING_DEVICES
        )

    def _get_task_timing_values(self, task) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Helper to calculate elapsed, ETR (currently None), and duration strings."""
        if not task or not hasattr(task, 'started_at') or task.started_at is None:
            is_ended_task = task and task.status in ['completed', 'failed']
            return None, None, "N/A" if is_ended_task else None

        started_at = task.started_at
        if isinstance(started_at, (int, float)):
            started_at = datetime.fromtimestamp(started_at, timezone.utc)
        elif hasattr(started_at, 'tzinfo') and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        elapsed_str_val, etr_str_val, duration_str_val = None, None, None

        if task.status in ['running', 'pending']:
            now = datetime.now(timezone.utc)
            elapsed_seconds = (now - started_at).total_seconds()
            if elapsed_seconds < 0: elapsed_seconds = 0
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_str_val = ""
            if hours > 0: elapsed_str_val += f"{int(hours)}h "
            if minutes > 0 or hours > 0: elapsed_str_val += f"{int(minutes)}m "
            elapsed_str_val += f"{int(seconds)}s"

        if task.status in ['completed', 'failed']:
            if hasattr(task, 'completed_at') and task.completed_at is not None:
                completed_at = task.completed_at
                if isinstance(completed_at, (int, float)):
                    completed_at = datetime.fromtimestamp(completed_at, timezone.utc)
                elif hasattr(completed_at, 'tzinfo') and completed_at.tzinfo is None:
                    completed_at = completed_at.replace(tzinfo=timezone.utc)

                if started_at and completed_at:
                    total_duration_seconds = (completed_at - started_at).total_seconds()
                    if total_duration_seconds < 0: total_duration_seconds = 0
                    hours, remainder = divmod(total_duration_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_str_val = ""
                    if hours > 0: duration_str_val += f"{int(hours)}h "
                    if minutes > 0 or hours > 0: duration_str_val += f"{int(minutes)}m "
                    duration_str_val += f"{int(seconds)}s"
                else:
                    duration_str_val = "N/A"
            else:
                duration_str_val = "N/A"

        if task.status not in ['running', 'pending'] and not duration_str_val:
            duration_str_val = "N/A"

        return elapsed_str_val, etr_str_val, duration_str_val

    def _get_task_timing_html(self, task, unique_prefix: str = "generic") -> str:
        """Generates HTML for displaying task timing information (elapsed/remaining)."""
        elapsed_str_val, etr_str_val, duration_str_val = self._get_task_timing_values(task)

        template = self.jinja_env.get_template("status_modal_parts/task_timing.html")
        return template.render(
            elapsed_str=elapsed_str_val,
            etr_str=etr_str_val,
            duration_str=duration_str_val,
            unique_prefix=unique_prefix
        )

    def _get_hardware_info_html(self) -> str:
        """Renders hardware info using a Jinja2 template."""
        num_cpus = os.cpu_count()
        allowed_db_workers = sum(IngestionConfig.MAX_DB_WORKERS_PER_TARGET.values())
        allowed_embedding_workers = AIConfig.MAX_EMBEDDING_WORKERS
        total_allowed_workers = allowed_db_workers + allowed_embedding_workers

        gpu_context = {
            "torch_available": torch is not None,
            "cuda_available": False,
            "cuda_version": None,
            "num_gpus": 0,
            "gpu_names": []
        }
        if gpu_context["torch_available"] and torch.cuda.is_available():
            gpu_context["cuda_available"] = True
            gpu_context["cuda_version"] = torch.version.cuda
            gpu_context["num_gpus"] = torch.cuda.device_count()
            gpu_context["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_context["num_gpus"])]

        cpu_load = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        context = {
            "num_cpus": num_cpus,
            "cpu_load": cpu_load,
            "ram_used_gb": ram.used / (1024**3),
            "ram_total_gb": ram.total / (1024**3),
            "ram_percent_used": ram.percent,
            "total_allowed_workers": total_allowed_workers,
            "db_workers_per_target": IngestionConfig.MAX_DB_WORKERS_PER_TARGET,
            "max_embedding_workers": AIConfig.MAX_EMBEDDING_WORKERS,
            **gpu_context
        }
        template = self.jinja_env.get_template("status_modal_parts/hardware_info.html")
        return template.render(context)

    def _get_storage_info_html(self) -> str:
        """Renders storage info using a Jinja2 template."""
        pg_size_bytes = self.framework.get_postgres_db_size()
        pg_size_str = "N/A"
        if pg_size_bytes is not None:
            if pg_size_bytes < 1024: pg_size_str = f"{pg_size_bytes} Bytes"
            elif pg_size_bytes < 1024**2: pg_size_str = f"{pg_size_bytes/1024:.2f} KB"
            elif pg_size_bytes < 1024**3: pg_size_str = f"{pg_size_bytes/1024**2:.2f} MB"
            else: pg_size_str = f"{pg_size_bytes/1024**3:.2f} GB"

        dgraph_usage_str = self.framework.get_dgraph_disk_usage() or "N/A"

        context = {
            "pg_db_name": PG_DB_NAME,
            "pg_size_str": pg_size_str,
            "dgraph_host": DGRAPH_HOST,
            "dgraph_port": DGRAPH_PORT,
            "dgraph_usage_str": dgraph_usage_str
        }
        template = self.jinja_env.get_template("status_modal_parts/storage_info.html")
        return template.render(context)

    def _get_usage_stats_html(self) -> str:
        """Renders usage stats using a Jinja2 template."""
        stats = self.framework.get_usage_statistics()
        template = self.jinja_env.get_template("status_modal_parts/usage_stats.html")
        return template.render(stats=stats)

    def _format_task_log_html(self, tasks: List[Task], existing_html: str = "") -> str:
        """Formats a list of tasks into an HTML string for the log using Jinja2 templates."""
        if not tasks and not existing_html:
            return "<div class='task-log-entry section-card'>No task history found.</div>" # Kept simple for no tasks
        if not tasks and existing_html:
             return existing_html

        html_parts = [existing_html] if existing_html and existing_html != "<div class='task-log-entry section-card'>No task history found.</div>" else []

        template = self.jinja_env.get_template("task_log_entry.html")

        for task in tasks:
            display_name = get_display_task_name(task.name)
            duration_str = ""
            # Ensure timing calculations use the original task object
            # The issue description "Timing N/A" for completed task in history needs to be addressed here
            # by ensuring _get_task_timing_values correctly calculates duration for completed tasks.
            # And that task.completed_at is properly set.

            # Let's re-check _get_task_timing_values for completed tasks.
            # It seems to calculate 'duration_str_val' correctly if task.completed_at is set.
            # The problem might be that task.completed_at is not always available or task object in history is partial.
            # The log shows "Timing N/A", "Details: Files Found: 55, Partitions Created: 2"
            # This implies the task object from history might be missing `completed_at` or `started_at`.
            # The `TaskRead` Pydantic model should include these.
            # `framework.get_task_history` should provide these.

            # For now, assume task object from history is complete.
            # The `_get_task_timing_values` function itself seems fine for calculating duration.
            # The issue might be in how `duration_str` is formed here or if `task.completed_at` is None.

            if task.status in ['completed', 'failed'] and task.started_at and task.completed_at:
                # Ensure timezone awareness for subtraction if not already
                started_at = task.started_at
                if hasattr(started_at, 'tzinfo') and started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)

                completed_at = task.completed_at
                if hasattr(completed_at, 'tzinfo') and completed_at.tzinfo is None:
                    completed_at = completed_at.replace(tzinfo=timezone.utc)

                duration_seconds = (completed_at - started_at).total_seconds()
                if duration_seconds < 0: duration_seconds = 0
                h, rem = divmod(duration_seconds, 3600)
                m, s = divmod(rem, 60)
                duration_parts = []
                if h > 0: duration_parts.append(f"{int(h)}h")
                if m > 0 or h > 0 : duration_parts.append(f"{int(m)}m") # show minutes if hours or minutes > 0
                duration_parts.append(f"{int(s)}s")
                duration_str = f" ({' '.join(duration_parts)})" if any(duration_parts) else ""
            elif task.status in ['running', 'pending'] and task.started_at:
                # Calculate elapsed time for running/pending tasks for history view if needed
                now = datetime.now(timezone.utc)
                elapsed_seconds = (now - (task.started_at.replace(tzinfo=timezone.utc) if task.started_at.tzinfo is None else task.started_at)).total_seconds()
                if elapsed_seconds < 0: elapsed_seconds = 0
                hours, remainder = divmod(elapsed_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed_str_val = ""
                if hours > 0: elapsed_str_val += f"{int(hours)}h "
                if minutes > 0 or hours > 0: elapsed_str_val += f"{int(minutes)}m "
                elapsed_str_val += f"{int(seconds)}s"
                duration_str = f" (Elapsed: {elapsed_str_val})"
            else: # Task is completed/failed but timing info is missing for some reason
                duration_str = " (Timing N/A)"


            html_parts.append(template.render(task=task, display_task_name=display_name, duration_str=duration_str))

        return "".join(html_parts)

    # handle_load_more_tasks method REMOVED

    def create_ui(self) -> gr.Blocks:
        """Builds the Gradio Blocks UI."""

        modal_js = """
        <script>
            function showModal(id) {
                const modal = document.getElementById(id);
                if (modal) modal.style.display = 'flex';
            }
            function hideModal(id) {
                const modal = document.getElementById(id);
                if (modal) modal.style.display = 'none';
                return null; // Required for Gradio JS event handlers
            }
            function setupModalEventListeners() {
                console.log("setupModalEventListeners called"); // DIAGNOSTIC
                // Listener for clicking on the modal background to close
                document.querySelectorAll('.modal-background').forEach(modalBg => {
                    modalBg.addEventListener('click', function(event) {
                        if (event.target === modalBg) { // Click was directly on the background
                            // Find the modal-content-wrapper or the modal ID itself
                            const modalContent = modalBg.querySelector('.modal-content-wrapper') || modalBg;
                            if (modalContent && modalContent.id) { // Ensure we have an ID to hide
                               hideModal(modalContent.id);
                            } else if (modalBg.id) { // Fallback to modalBg's ID
                               hideModal(modalBg.id);
                            }
                        }
                    });
                });

                // Global ESC key listener
                document.addEventListener('keydown', function(event) {
                    console.log("Keydown event:", event.key); // DIAGNOSTIC
                    if (event.key === 'Escape') {
                        console.log("Escape key pressed."); // DIAGNOSTIC
                        const selector = '.modal-background[style*="display: flex"], .modal-background[style*="display:block"]';
                        const visibleModals = document.querySelectorAll(selector);
                        console.log("Visible modals query selector:", selector, "Found:", visibleModals); // DIAGNOSTIC

                        visibleModals.forEach(modal => {
                            console.log("Processing visible modal:", modal.id); // DIAGNOSTIC
                            // The modal element selected by querySelectorAll already has the ID (e.g., "addRepoModal").
                            // So, modal.id should be directly usable.
                            if (modal.id) {
                                hideModal(modal.id);
                            } else {
                                // This case should ideally not happen if modals are structured with IDs on .modal-background
                                console.warn("ESC key: Found a visible modal background without an ID.", modal);
                            }
                        });
                    }
                });
            }
            // Run setup after Gradio loads its components
            if (window.gradioApp && typeof window.gradioApp.addEventListener === 'function') {
                 window.gradioApp.addEventListener('render', setupModalEventListeners);
            } else {
                 // Fallback for older Gradio or if gradioApp isn't fully ready
                 document.addEventListener('DOMContentLoaded', setupModalEventListeners);
            }

            // Prevent flicker on progress bar updates
            window.addEventListener('DOMContentLoaded', function() {
                const progressBar = document.querySelector('#main-progress-bar');
                if (progressBar) {
                    progressBar.style.transition = 'none';
                    setTimeout(() => {
                        progressBar.style.transition = '';
                    }, 100);
                }
            });
        </script>
        """

        fontawesome_cdn = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">'
        tailwind_cdn = '<script src="https://cdn.tailwindcss.com"></script>'
        dynamic_updates_js_link = '<script src="/static/js/dynamic_updates.js"></script>'
        ast_viewer_js_link = '<script src="/static/js/ast_viewer.js"></script>' # Link for the new JS file


        # Load CSS from file
        css_file_path = Path(__file__).parent / "static" / "css" / "control_panel.css"
        try:
            with open(css_file_path, "r") as f:
                control_panel_css = f.read()
        except FileNotFoundError:
            control_panel_css = "/* CSS file not found. Styles will be missing. */"
            print(f"Warning: CSS file not found at {css_file_path}")

        app_head_content = tailwind_cdn + modal_js + fontawesome_cdn + dynamic_updates_js_link + ast_viewer_js_link
        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="sky"), title="SDA Framework", css=control_panel_css, head=app_head_content) as demo:
            gr.Markdown("# Software Development Analytics")
            with gr.Row(elem_classes="control-button-row sda-status-bar-row"): # Added sda-status-bar-row for CSS targeting
                status_output = gr.Textbox(interactive=False, placeholder="Status messages will appear here...", scale=4, lines=1, show_label=False, container=False) # Removed label, added lines=1, show_label=False, container=False
                # view_status_modal_btn = gr.Button("View Control Panel", scale=1) # Original button
                # Attempting HTML button with icon for "View Control Panel"
                view_status_modal_btn_html = gr.HTML(
                    "<button class='gr-button gr-button-lg gr-button-secondary button-with-icon-content' onclick=\"showModal('statusModal')\" title='View Control Panel'>" + \
                    "<i class='fas fa-tasks mr-1.5'></i>View Control Panel" + \
                    "</button>",
                    elem_classes="html-button-wrapper" # Class for the wrapper div Gradio creates
                )

            # Redundant progress_row and main_progress_bar REMOVED

            repo_id_state = gr.State()
            branch_state = gr.State()
            selected_file_state = gr.State()
            last_status_text_state = gr.State("") # Still used for overall status text, not modal HTML
            # last_progress_html_state REMOVED
            # task_log_offset_state REMOVED
            # current_task_log_html_state REMOVED
            # js_update_data_output = gr.JSON(visible=False, label="JS Update Data", elem_id="js_update_data_json") # REMOVED
            # States for tracking Control Panel structure - REMOVED as modal is now self-contained
            # last_main_task_id_state = gr.State(None)
            # last_sub_task_ids_state = gr.State([])
            # last_main_task_has_details_state = gr.State(False)
            # last_main_task_has_error_state = gr.State(False)
            # ast_trigger_textbox removed


            with gr.Column(elem_id="addRepoModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    with gr.Row(elem_classes="modal-header"):
                        gr.Markdown("## Add New Repository", elem_classes="modal-title")
                        gr.HTML('<button title="Close" class="modal-close-x" onclick="hideModal(\'addRepoModal\')"><i class="fas fa-times"></i></button>', elem_classes="modal-close-x-wrapper")
                    repo_url_modal = gr.Textbox(label="Git Repository URL or Local Path", placeholder="https://github.com/user/repo.git or /path/to/local/repo")
                    with gr.Row():
                        add_repo_submit_btn = gr.Button("Add & Analyze", variant="primary")
                        add_repo_cancel_btn = gr.Button("Cancel") # This button also calls hideModal via its click event later

            with gr.Column(elem_id="codeViewerModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    with gr.Row(elem_classes="modal-header"):
                        gr.Markdown("## File Content", elem_classes="modal-title") # Title for code viewer
                        gr.HTML('<button title="Close" class="modal-close-x" onclick="hideModal(\'codeViewerModal\')"><i class="fas fa-times"></i></button>', elem_classes="modal-close-x-wrapper")
                    modal_code_viewer = gr.Code(label="File Content", language=None, interactive=False)
                    modal_close_btn = gr.Button("Close") # This button also calls hideModal

            with gr.Column(elem_id="statusModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    with gr.Row(elem_classes="modal-header"):
                        gr.Markdown("## Control Panel", elem_classes="modal-title")
                        gr.HTML('<button title="Close" class="modal-close-x" onclick="hideModal(\'statusModal\')"><i class="fas fa-times"></i></button>', elem_classes="modal-close-x-wrapper")
                    # Control Panel is now an iframe loading the standalone HTML page
                    status_details_html = gr.HTML(
                        value='<iframe src="/static/control_panel.html" style="width: 100%; height: 70vh; border: none;"></iframe>',
                        elem_id="control_panel_iframe_container_wrapper"
                    )
                    status_modal_close_btn = gr.Button("Close")

            with gr.Tabs() as tabs:
                with gr.TabItem(label="🗂️ Repository & Agent", id=0):
                    with gr.Row():
                        with gr.Column(scale=1):
                            repo_dropdown = gr.Dropdown(label="Select Repository", interactive=True)
                            open_add_repo_modal_btn = gr.Button("Add New Repository")
                        with gr.Column(scale=1):
                            branch_dropdown = gr.Dropdown(label="Select Branch", interactive=True)
                            analyze_branch_btn = gr.Button("Force Re-Analyze Branch", variant="primary")

                    chatbot = gr.Chatbot(
                        label="Chat with your Codebase", height=600,
                        avatar_images=(str(Path("assets/user.png").resolve()), str(Path("assets/bot.png").resolve())),
                        show_copy_button=True, type="messages"
                    )

                    gr.ChatInterface(
                        fn=self.handle_chat_stream, chatbot=chatbot,
                        textbox=gr.Textbox(placeholder="Ask a question about the repository...", container=False, scale=7, lines=3),
                        additional_inputs=[repo_id_state, branch_state]
                    )

                with gr.TabItem(label="📊 Insights Dashboard", id=1):
                    with gr.Row():
                        with gr.Column(scale=1): # This column will now hold the HTML stat cards
                            gr.Markdown('### <i class="fas fa-chart-bar mr-1.5"></i> Key Statistics') # Icons in Markdown are OK
                            stats_cards_html = gr.HTML(label="Key Repository Statistics")
                        with gr.Column(scale=1):
                            gr.Markdown('### <i class="fas fa-code mr-1.5"></i> Language Breakdown') # Icons in Markdown are OK
                            lang_plot = gr.Plot(label="Language Breakdown") # This remains a plot
                    gr.Markdown('### <i class="fas fa-search-plus mr-1.5"></i> In-Depth Analysis (runs in background)') # Icons in Markdown are OK
                    with gr.Row():
                        analyze_dead_code_btn = gr.Button("Find Unused Code")
                        analyze_duplicates_btn = gr.Button("Find Duplicate Code")
                    # Full Task History Accordion REMOVED
                    with gr.Tabs():
                        with gr.TabItem("Unused Code Results"):
                            dead_code_df = gr.DataFrame(headers=["File", "Symbol", "Type", "Lines"], interactive=False, max_height=400,
                                                        datatype=["str", "str", "str", "str"])
                        with gr.TabItem("Duplicate Code Results"):
                            duplicate_code_df = gr.DataFrame(headers=["File A", "Lines A", "File B", "Lines B", "Similarity"], interactive=False,
                                                             max_height=400)

                with gr.TabItem(label="📄 Document Comprehension", id=2):
                    with gr.Row():
                        with gr.Column(scale=1): # File Explorer Column
                            gr.Markdown("#### File Explorer")
                            from sda.config import WORKSPACE_DIR as SDA_WORKSPACE_DIR
                            file_explorer = gr.FileExplorer(
                                root_dir=str(SDA_WORKSPACE_DIR),
                                glob="**/*",
                                label="Repository Files",
                                interactive=True,
                                file_count="single",
                                elem_id="sda_file_explorer"
                            )
                        with gr.Column(scale=3): # Content Column
                            with gr.Tabs() as content_tabs:
                                with gr.TabItem(label="🔗 Embedding", id="embedding_tab"):
                                    embedding_html_viewer = gr.HTML(label="Node Embedding Visualization", elem_id="ast_iframe_container_wrapper")
                                    # Placeholder for actual content
                                with gr.TabItem(label="⇆ Change Analysis", id="change_analysis_tab"):
                                    with gr.Accordion("Analyze Current Changes for Selected File", open=True): # Title updated
                                        gr.Markdown("Analyzes uncommitted changes for the file selected in the File Explorer (vs. current branch HEAD).")
                                        # current_modified_files_dropdown_ca REMOVED from here
                                        analyze_selected_file_current_changes_btn = gr.Button("Analyze Current Changes for Selected File", variant="secondary") # Renamed button

                                    with gr.Accordion("Compare Selected File with its Older Version", open=False): # Title emphasizes selected file
                                        gr.Markdown("Compare the file selected in File Explorer (from current branch HEAD) with an older version of itself.")
                                        # file_to_compare_dropdown_ca was already removed conceptually.

                                        with gr.Row():
                                            branch_compare_older_version_ca = gr.Dropdown(label="Branch of Older Version", interactive=True, info="Select branch of the older version.")
                                            commit_older_version_ca = gr.Textbox(label="Older Version (Commit SHA/Tag)", placeholder="HEAD or commit SHA...", scale=1)

                                        compare_with_older_btn = gr.Button("Analyze Difference with Older Version", variant="secondary")

                                    change_analysis_output = gr.Markdown("LLM analysis of changes will appear here.")

                                with gr.TabItem(label="↔️ Raw Diff", id="raw_diff_tab"):
                                    no_changes_message_html = gr.HTML(
                                        "<div style='font-size: 1.5em; text-align: center; padding: 40px; color: #888;'>NO CHANGES</div>",
                                        visible=False,
                                        elem_id="raw_diff_no_changes_message"
                                    )
                                    code_viewer = gr.Code(label="Raw Diff Output", language=None, interactive=False, visible=True)
                                    image_viewer = gr.Image(label="Image Content", interactive=False, visible=False)

            self.task_buttons = [analyze_branch_btn, analyze_dead_code_btn, analyze_duplicates_btn, add_repo_submit_btn,
                                 analyze_selected_file_current_changes_btn, compare_with_older_btn] # Added new CA buttons
            timer = gr.Timer(2)

            all_insight_outputs = [stats_cards_html, lang_plot]
            doc_comp_outputs = [ # Updated to 8 items
                file_explorer,
                embedding_html_viewer,
                change_analysis_output,
                code_viewer,
                image_viewer,
                selected_file_state,
                branch_compare_older_version_ca,
                no_changes_message_html # Added new HTML component
            ]


            # Define these components early if they are needed by reference in poll_outputs
            # For this specific fix, we will pass them to handle_polling and it will return gr.update() for them.
            # The actual Gradio component instances are repo_dropdown, branch_dropdown.

            # Initial load of task log (first page) when UI loads or repo changes
            # We need a way to trigger this. For now, let's make it part of handle_repo_select
            # and handle_initial_load.

            # The task_log_output is now full_task_log_html
            demo.load(
                self.handle_initial_load,
                outputs=[repo_dropdown, branch_dropdown, repo_id_state, branch_state, chatbot]
            ).then(
                self.update_all_panels,
                [repo_id_state, branch_state], # No current_path_state needed for update_all_panels now
                all_insight_outputs + doc_comp_outputs
            )


            # Poll inputs updated: modal-specific states removed.
            poll_inputs = [
                repo_id_state, branch_state,
                last_status_text_state     # For overall status message logic
            ]

            # Poll outputs updated: components and states for the old modal update mechanism are removed.
            poll_outputs = [
                status_output,                      # Overall status message
                dead_code_df, duplicate_code_df,    # Dataframe updates
                stats_cards_html, lang_plot,        # Updated: stats_plot to stats_cards_html
                last_status_text_state,             # Pass-through state for overall status
                branch_dropdown, branch_state       # Branch updates
            ] + self.task_buttons
            timer.tick(self.handle_polling, poll_inputs, poll_outputs)


            open_add_repo_modal_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'flex'; }")
            add_repo_cancel_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }")
            modal_close_btn.click(None, js="() => { const modal = document.getElementById('codeViewerModal'); if (modal) modal.style.display = 'none'; }")
            status_modal_close_btn.click(None, js="() => { const modal = document.getElementById('statusModal'); if (modal) modal.style.display = 'none'; }")

            add_repo_submit_btn.click(
                self.handle_add_repo, [repo_url_modal], [status_output, repo_dropdown] + self.task_buttons
            ).then(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }").then(
                self.handle_repo_select, [repo_dropdown],
                [branch_dropdown, repo_id_state, branch_state, chatbot]
            )

            repo_dropdown.change(
                self.handle_repo_select, [repo_dropdown],
                [branch_dropdown, repo_id_state, branch_state, chatbot]
            ).then(
                self.update_all_panels,
                [repo_id_state, branch_state], # No current_path_state
                all_insight_outputs + doc_comp_outputs
            )

            branch_dropdown.change(
                self.handle_branch_select, [branch_dropdown],
                [branch_state, chatbot]
            ).then(
                self.update_all_panels,
                [repo_id_state, branch_state], # No current_path_state
                all_insight_outputs + doc_comp_outputs
            )

            analyze_branch_btn.click(self.handle_analyze_branch, [repo_id_state, branch_state], [status_output] + self.task_buttons)
            analyze_dead_code_btn.click(self.handle_run_dead_code, [repo_id_state, branch_state], [status_output] + self.task_buttons)
            analyze_duplicates_btn.click(self.handle_run_duplicates, [repo_id_state, branch_state], [status_output] + self.task_buttons)

            dead_code_df.select(self.handle_code_item_select, [repo_id_state, branch_state, dead_code_df], [modal_code_viewer]).then(
                None, js="() => { const modal = document.getElementById('codeViewerModal'); if (modal) modal.style.display = 'flex'; }")

            # New FileExplorer change event
            file_explorer.change( # Changed from .select to .change
                self.handle_file_explorer_select,
                inputs=[repo_id_state, branch_state, file_explorer], # Pass file_explorer itself as input for its value
                outputs=[
                    embedding_html_viewer, code_viewer, image_viewer, selected_file_state,
                    no_changes_message_html # ast_trigger_textbox and .then() call removed
                    # current_modified_files_dropdown_ca, file_to_compare_dropdown_ca REMOVED
                ]
            ) # .then() call removed

            content_tabs.select(
                self.handle_content_tab_select,
                inputs=[repo_id_state, branch_state, selected_file_state],
                outputs=[change_analysis_output]
            )

            # Connect Change Analysis buttons
            # analyze_current_changes_btn was renamed to analyze_selected_file_current_changes_btn
            # and its input current_modified_files_dropdown_ca was removed.
            # It now uses selected_file_state.
            analyze_selected_file_current_changes_btn.click(
                self.handle_analyze_current_file_changes,
                inputs=[repo_id_state, branch_state, selected_file_state], # Uses selected_file_state
                outputs=[change_analysis_output]
            )
            compare_with_older_btn.click( # Renamed from compare_versions_btn
                self.handle_analyze_version_comparison,
                inputs=[ # Updated inputs
                    repo_id_state, branch_state, selected_file_state, # current branch and file
                    branch_compare_older_version_ca, commit_older_version_ca # older version details
                ],
                outputs=[change_analysis_output]
            )

            # Removed event handlers for modified_files_dropdown_diff, revert_file_btn_diff, commit_btn_diff

        return demo

    def handle_chat_stream(self, message: str, history: List[Dict[str, str]], repo_id: int, branch: str) -> Generator[List[Dict[str, str]], None, None]:
        """
        Handles streaming chat responses. Yields the entire updated history object
        at each step to be compatible with Gradio's `type="messages"` format.
        """
        if not repo_id or not branch:
            history.append({"role": "assistant", "content": "Please select a repository and branch first."})
            yield history
            return

        # The history passed to the agent should not include the latest user message,
        # as that is the new query. LlamaIndex handles this separation.
        llama_index_history = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in history[:-1]]

        response_generator = self.framework.run_agent_chat_stream(
            query=message,
            repo_id=repo_id,
            branch=branch,
            chat_history=llama_index_history
        )

        # Add a placeholder for the assistant's response to the history that will be displayed
        history.append({"role": "assistant", "content": ""})

        full_response = ""
        for chunk in response_generator:
            full_response = chunk
            # Update the content of the assistant's message placeholder
            history[-1]["content"] = full_response
            # Yield the entire, correctly formatted history object
            yield history

    def handle_polling(
        self, repo_id: int, branch: str,
        last_status_text: str # For overall status message
        # last_progress_html: str REMOVED
    ) -> Tuple:
        branch_dropdown_update = gr.skip() # Initialize to skip update by default
        branch_state_update = gr.update() # State updates are usually fine

        # Initialize new state outputs - these might be removed if not used by other components
        # new_last_main_task_id = _last_main_task_id
        # new_last_sub_task_ids = list(_last_sub_task_ids)
        # new_last_main_task_has_details = _last_main_task_has_details
        # new_last_main_task_has_error = _last_main_task_has_error

        # This data structure will be sent over WebSocket
        control_panel_ws_data = {
            "main_task": None, "sub_tasks": [], "hardware_info": None,
            # We can also include the external progress bar data here if the main HTML page wants to show it too
            # or keep it separate if only the old Gradio main_progress_bar needs it.
            # For full decoupling, the new HTML page should handle its own version of external progress bar.
            # "external_progress_bar_data" REMOVED from WebSocket data
            "current_repo_id": repo_id, # Added current_repo_id for JS
            "system_info": { # For model, storage, usage stats
                "model_info": {},
                "storage_info": {},
                "usage_stats": {}
            }
        }

        # Populate system_info (can be done once or less frequently if these are static)
        control_panel_ws_data["system_info"]["model_info"] = {
            "active_llm_model": AIConfig.ACTIVE_LLM_MODEL,
            "active_embedding_model": AIConfig.ACTIVE_EMBEDDING_MODEL,
            "embedding_devices": AIConfig.EMBEDDING_DEVICES
        }
        pg_size_bytes = self.framework.get_postgres_db_size()
        pg_size_str = "N/A"
        if pg_size_bytes is not None:
            if pg_size_bytes < 1024: pg_size_str = f"{pg_size_bytes} Bytes"
            elif pg_size_bytes < 1024**2: pg_size_str = f"{pg_size_bytes/1024:.2f} KB"
            elif pg_size_bytes < 1024**3: pg_size_str = f"{pg_size_bytes/1024**2:.2f} MB"
            else: pg_size_str = f"{pg_size_bytes/1024**3:.2f} GB"
        control_panel_ws_data["system_info"]["storage_info"] = {
            "pg_db_name": PG_DB_NAME,
            "pg_size_str": pg_size_str,
            "dgraph_host": DGRAPH_HOST,
            "dgraph_port": DGRAPH_PORT, # Note: HTML uses this to form host:port string
            "dgraph_usage_str": self.framework.get_dgraph_disk_usage() or "N/A"
        }
        control_panel_ws_data["system_info"]["usage_stats"] = self.framework.get_usage_statistics()


        current_cpu_load = psutil.cpu_percent(interval=None)
        current_ram_percent = psutil.virtual_memory().percent
        current_ram_used_gb = psutil.virtual_memory().used / (1024**3)
        current_ram_total_gb = psutil.virtual_memory().total / (1024**3)
        control_panel_ws_data["hardware_info"] = {
            "cpu_load": current_cpu_load,
            "ram_percent": current_ram_percent,
            "ram_absolute_text": f"{current_ram_used_gb:.1f} / {current_ram_total_gb:.1f} GB",
            "num_cpus": os.cpu_count(),
            "total_allowed_workers": sum(IngestionConfig.MAX_DB_WORKERS_PER_TARGET.values()) + AIConfig.MAX_EMBEDDING_WORKERS,
            "db_workers_per_target": IngestionConfig.MAX_DB_WORKERS_PER_TARGET,
            "max_embedding_workers": AIConfig.MAX_EMBEDDING_WORKERS,
            "gpu_info": {
                "torch_available": torch is not None,
                "cuda_available": False,
                "cuda_version": None,
                "num_gpus": 0,
                "gpu_names": []
            }
        }
        if control_panel_ws_data["hardware_info"]["gpu_info"]["torch_available"] and torch.cuda.is_available():
            control_panel_ws_data["hardware_info"]["gpu_info"]["cuda_available"] = True
            control_panel_ws_data["hardware_info"]["gpu_info"]["cuda_version"] = torch.version.cuda
            control_panel_ws_data["hardware_info"]["gpu_info"]["num_gpus"] = torch.cuda.device_count()
            control_panel_ws_data["hardware_info"]["gpu_info"]["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        # The status_details_html and its related states (_last_main_task_id etc.) are no longer updated here for Gradio.
        # All Control Panel modal updates go via WebSocket.

        if not repo_id:
            # control_panel_ws_data["external_progress_bar_data"]["message"] = "No repository selected" # REMOVED
            control_panel_ws_data["main_task"] = None
            # default_ext_progress_html = self._create_html_progress_bar(0, "No repository selected", "Idle", unique_prefix="external") # REMOVED

            if main_event_loop:
                asyncio.run_coroutine_threadsafe(control_panel_manager.broadcast(control_panel_ws_data), main_event_loop)
            else:
                print("Error: Main event loop not available for WebSocket broadcast.")


            return (
                "No repository selected.", # status_output
                # status_details_html output removed
                gr.update(), # dead_code_df
                gr.update(), # duplicate_code_df
                gr.skip(), # stats_plot (stats_cards_html)
                gr.skip(), # lang_plot
                last_status_text, # last_status_text_state (pass-through)
                # Progress bar updates REMOVED
                branch_dropdown_update,
                branch_state_update
            ) + self._get_task_button_updates(True)

        task = self.framework.get_latest_task(repo_id)

        if not task:
            # control_panel_ws_data["external_progress_bar_data"]["message"] = "No active tasks" # REMOVED
            control_panel_ws_data["main_task"] = None
            # default_ext_progress_html = self._create_html_progress_bar(0, "No active tasks", "Idle", unique_prefix="external") # REMOVED

            if main_event_loop:
                asyncio.run_coroutine_threadsafe(control_panel_manager.broadcast(control_panel_ws_data), main_event_loop)
            else:
                print("Error: Main event loop not available for WebSocket broadcast.")

            return (
                "No tasks found for this repository.", # status_output
                gr.update(), gr.update(), # dead_code_df, duplicate_code_df
                gr.skip(), # stats_plot (stats_cards_html)
                gr.skip(), # lang_plot
                last_status_text, # last_status_text_state
                # Progress bar updates REMOVED
                branch_dropdown_update,
                branch_state_update,
            ) + self._get_task_button_updates(True)

        # --- Task exists, populate control_panel_ws_data ---
        display_task_name = get_display_task_name(task.name)
        status_msg = f"Task '{display_task_name}': {task.message} ({task.progress:.0f}%)"
        is_running = task.status in ['running', 'pending']

        # control_panel_ws_data["external_progress_bar_data"] entries REMOVED

        elapsed_str, _, duration_str = self._get_task_timing_values(task)

        main_task_status_classes_map = {
            'running': "bg-blue-200 text-blue-800 dark:bg-blue-700 dark:text-blue-200",
            'completed': "bg-green-200 text-green-800 dark:bg-green-700 dark:text-green-200",
            'failed': "bg-red-200 text-red-800 dark:bg-red-700 dark:text-red-200",
            'pending': "bg-yellow-200 text-yellow-800 dark:bg-yellow-700 dark:text-yellow-200"
        }
        main_task_status_class = main_task_status_classes_map.get(task.status, "bg-gray-200 text-gray-800 dark:bg-gray-600 dark:text-gray-200")
        main_task_status_class += " px-3 py-1 text-xs font-semibold rounded-full"

        control_panel_ws_data["main_task"] = {
            "id": task.id, # Added task ID
            "name": display_task_name, "status_text": task.status, "status_class": main_task_status_class,
            "progress": task.progress, "message": task.message,
            "time_elapsed": elapsed_str, "time_duration": duration_str,
            "details": task.details if task.details else {}, # Ensure details are always a dict
            "error_message": task.error_message,
            # Children will be added below
        }

        # current_sub_task_ids = sorted([st.id for st in task.children]) if task.children else [] # Not directly needed for Gradio return
        # current_main_task_has_details = bool(task.details) # Not directly needed for Gradio return
        # current_main_task_has_error = bool(task.error_message) # Not directly needed for Gradio return

        # new_last_main_task_id = task.id # These states are not part of the return tuple for Gradio anymore
        # new_last_sub_task_ids = current_sub_task_ids
        # new_last_main_task_has_details = current_main_task_has_details
        # new_last_main_task_has_error = current_main_task_has_error

        # Prepare children data to be nested if task exists
        children_list_for_main_task = []
        if task and task.children:
            sub_task_status_classes_map = {
                'running': "bg-blue-100 text-blue-700 dark:bg-blue-600 dark:text-blue-100",
                'completed': "bg-green-100 text-green-700 dark:bg-green-600 dark:text-green-100",
                'failed': "bg-red-100 text-red-700 dark:bg-red-600 dark:text-red-100",
                'pending': "bg-yellow-100 text-yellow-700 dark:bg-yellow-600 dark:text-yellow-100"
            }
            default_sub_task_class_base = "bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-100"
            sub_task_badge_common_classes = " text-xxs px-1.5 py-0.5 rounded-full whitespace-nowrap flex-shrink-0"

            for child_task_obj in sorted(task.children, key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc)):
                child_display_name = get_display_task_name(child_task_obj.name)
                child_status_class = sub_task_status_classes_map.get(child_task_obj.status, default_sub_task_class_base) + sub_task_badge_common_classes
                children_list_for_main_task.append({
                    "id": child_task_obj.id,
                    "name": child_display_name,
                    "status_text": child_task_obj.status,
                    "status_class": child_status_class,
                    "progress": child_task_obj.progress,
                    "message": child_task_obj.message,
                    "details": child_task_obj.details if child_task_obj.details else {}, # Ensure details are always a dict
                    # Note: error_message for subtasks is not explicitly handled in current JS template, but can be added
                    "error_message": child_task_obj.error_message
                })

        # Update main_task in WebSocket data, now including children
        if task: # task is the result of self.framework.get_latest_task(repo_id)
            control_panel_ws_data["main_task"]["children"] = children_list_for_main_task
        # The top-level "sub_tasks" key in control_panel_ws_data is no longer populated or needed.

        if main_event_loop:
            asyncio.run_coroutine_threadsafe(control_panel_manager.broadcast(control_panel_ws_data), main_event_loop)
        else:
            print("Error: Main event loop not available for WebSocket broadcast.")

        # Prepare updates for Gradio components (excluding the modal)
        # current_ext_progress_html and main_progress_update REMOVED
        # progress_row_update REMOVED

        button_updates = self._get_task_button_updates(interactive=not is_running)
        dead_code_update, dup_code_update = gr.update(), gr.update()
        # Initialize updates for HTML stats and language plot to "skip update" by default
        stats_html_update, lang_plot_update = gr.skip(), gr.skip()

        task_could_change_branch = task.name.startswith("Ingest Branch:") or task.name == "analyze_branch" # Use internal name for logic
        if task.status == 'completed':
            status_msg = f"Last task '{display_task_name}' completed successfully." # Use display_task_name for UI
            if task.result: # Check internal name for result processing
                if task.name == "find_dead_code" and task.result.get("dead_code"):
                    dead_code_data = task.result["dead_code"]
                    df_data = [[dc.get('file_path'), dc.get('name'), dc.get('node_type'), f"{dc.get('start_line')}-{dc.get('end_line')}"] for dc in dead_code_data]
                    dead_code_update = pd.DataFrame(df_data, columns=["File", "Symbol", "Type", "Lines"])
                elif task.name == "find_duplicate_code" and task.result.get("duplicate_code"):
                    dup_data = task.result["duplicate_code"]
                    df_data = [[d.get('file_a'), d.get('lines_a'), d.get('file_b'), d.get('lines_b'), f"{d.get('similarity', 0):.2%}"] for d in dup_data]
                    dup_code_update = pd.DataFrame(df_data, columns=["File A", "Lines A", "File B", "Lines B", "Similarity"])

            if task_could_change_branch: # Logic based on internal name
                current_repo_branches = self.framework.get_repository_branches(repo_id)
                repo = self.framework.get_repository_by_id(repo_id)
                new_active_branch = repo.active_branch if repo else None
                if new_active_branch and new_active_branch not in current_repo_branches: # Check if active branch is valid
                    new_active_branch = current_repo_branches[0] if current_repo_branches else None

                # Ensure new_active_branch is not None before updating insights
                target_branch_for_insights = new_active_branch if new_active_branch else branch

                branch_dropdown_update = gr.update(choices=current_repo_branches, value=target_branch_for_insights)
                if branch != target_branch_for_insights : branch_state_update = target_branch_for_insights

                # Update insight plots
                s_html, l_plot = self.update_insights_dashboard(repo_id, target_branch_for_insights)
                stats_html_update = gr.update(value=s_html) # s_html is now the HTML string
                lang_plot_update = gr.update(value=l_plot)


        elif task.status == 'failed':
            status_msg = f"Task '{display_task_name}' Failed: Check logs for details." # Use display_task_name
            # Potentially clear or leave stale stats/plots as is, or show an error state
            # For now, they will retain their last state or be None if initialized that way.

        return (
            status_msg, # status_output
            # status_details_html output removed
            dead_code_update, dup_code_update, stats_html_update, lang_plot_update, # df_updates and plot_updates
            last_status_text, # last_status_text_state (pass through)
            # main_progress_update, REMOVED
            # progress_row_update, REMOVED
            # last_progress_html_state update REMOVED
            branch_dropdown_update, branch_state_update
            # js_update_data_output and modal states removed
        ) + button_updates

    def handle_initial_load(self) -> Tuple[gr.update, gr.update, Optional[int], Optional[str], List[Dict[str, str]]]:
        # Ensure task details are initialized properly.
        # The control panel uses WebSocket, so initial empty state is handled by JS client.
        # This function primarily sets up repo/branch dropdowns.
        repos = self.framework.get_all_repositories()
        repo_choices = [(f"{repo.name} ({repo.path})", repo.id) for repo in repos]
        initial_repo_id = repo_choices[0][1] if repo_choices else None
        repo_upd = gr.update(choices=repo_choices, value=initial_repo_id)

        # Outputs for handle_initial_load are:
        # repo_dropdown, branch_dropdown, repo_id_state, branch_state, chatbot
        if not initial_repo_id:
            return repo_upd, gr.update(choices=[], value=None), None, None, [{"role": "assistant", "content": "Welcome! Please add a repository to begin."}]

        # Call handle_repo_select to get most of the values
        branch_upd, repo_id_val, branch_val, chatbot_val = self.handle_repo_select(initial_repo_id) # type: ignore
        return repo_upd, branch_upd, repo_id_val, branch_val, chatbot_val


    def handle_add_repo(self, repo_identifier: str) -> Tuple:
        if not repo_identifier:
            return ("Please provide a repository URL or local path.", gr.update()) + self._get_task_button_updates(True)
        repo = self.framework.add_repository(repo_identifier)
        if not repo:
            return ("Failed to add repository.", gr.update()) + self._get_task_button_updates(True)
        repos = self.framework.get_all_repositories()
        choices = [(f"{r.name} ({r.path})", r.id) for r in repos]
        new_repo_update = gr.update(choices=choices, value=repo.id)
        if repo.active_branch:
            self.framework.analyze_branch(repo.id, repo.active_branch, 'user')
            status_msg = f"Added '{repo.name}'. Analysis for branch '{repo.active_branch}' is running."
            return (status_msg, new_repo_update) + self._get_task_button_updates(False)
        else:
            status_msg = f"Added '{repo.name}'. Please select a branch to analyze."
            return (status_msg, new_repo_update) + self._get_task_button_updates(True)

    def handle_repo_select(self, repo_id: int) -> Tuple[gr.update, Optional[int], Optional[str], List[Dict[str,str]]]:
        # Returns: branch_dropdown_update, repo_id_state, branch_state, chatbot_update
        if not repo_id:
            return gr.update(choices=[], value=None), None, None, [{"role": "assistant", "content": "Please select a repository."}]
        repo = self.framework.get_repository_by_id(int(repo_id))
        if not repo:
            return gr.update(choices=[], value=None), repo_id, None, []
        branches = self.framework.get_repository_branches(repo_id)
        active_branch = repo.active_branch if repo.active_branch in branches else (branches[0] if branches else None)
        chatbot_reset = [{"role": "assistant", "content": f"Agent ready for '{repo.name}' on branch '{active_branch}'."}]
        return gr.update(choices=branches, value=active_branch), repo_id, active_branch, chatbot_reset

    def handle_branch_select(self, branch: str) -> Tuple[str, List[Dict[str,str]]]:
        # Returns: branch_state, chatbot_update
        chatbot_msg = [{"role": "assistant", "content": f"Agent context switched to branch '{branch}'."}]
        return branch, chatbot_msg

    def handle_populate_file_explorer(self, repo_id: int, branch: str) -> gr.update:
        if not repo_id or not branch:
            print(f"Debug: handle_populate_file_explorer called with no repo_id ({repo_id}) or branch ({branch})")
            return gr.update(value=[])

        print(f"Debug: Populating file explorer for repo {repo_id}, branch {branch} by calling framework.get_file_tree")
        tree_data = self.framework.get_file_tree(repo_id, branch) # Calls the framework method

        # Ensure tree_data is in the format List[Tuple[str, str]] or List[Any] for Gradio Tree.
        # The framework.get_file_tree placeholder already returns this format.
        # This method is no longer used directly by update_all_panels for the main file browser.
        # It's still used by handle_populate_change_analysis_inputs.
        return gr.update(value=tree_data)

    # handle_populate_file_browser_radio, handle_file_browser_go_up, handle_file_browser_radio_select are removed.
    # New handlers for gr.FileExplorer will be simpler.

    def handle_file_explorer_select(self, repo_id: int, branch: str, selection_event_data: Any) -> Tuple[gr.update, gr.update, gr.update, str, gr.update]:
        # Returns updates for: embedding_html_viewer, code_viewer, image_viewer, selected_file_state, no_changes_message_html
        # selection_event_data is the value from FileExplorer's .change() event.
        # If file_count="single", this should be a single path string when a user selects a file.
        # If it's a list, it might be the initial population or a multi-select scenario we want to ignore for single-file processing.

        file_path: Optional[str] = None
        if isinstance(selection_event_data, str):
            file_path = selection_event_data
        elif isinstance(selection_event_data, list):
            # If file_count="single" is truly effective, this case might only occur if the component's value
            # (the whole list of files) is updated programmatically, triggering .change.
            # In such a case, we don't have a single user-selected file to process.
            logging.info(f"FileExplorer .change event triggered with a list (count: {len(selection_event_data)}). Assuming no single file selection by user. Skipping detailed view updates.")
            # We still need to return the correct number of updates.
            return gr.skip(), gr.skip(), gr.skip(), "" # Removed 2 gr.skip()

        if not file_path: # Path could be None if deselected or if it was a list and we decided to skip.
            embedding_html_update = gr.update(value="<div>Select a file for embedding visualization.</div>")
            code_viewer_update = gr.update(value="// Select a file to view content/diff.", language=None, label="File Content / Diff", visible=True)
            image_viewer_update = gr.update(value=None, visible=False)
            no_changes_html_update = gr.update(visible=False)
            return embedding_html_update, code_viewer_update, image_viewer_update, "", no_changes_html_update

        print(f"UI: FileExplorer selection changed to: '{file_path}' for repo {repo_id}, branch {branch}")

        # Convert absolute path from FileExplorer to relative path for framework
        repo = self.framework.get_repository_by_id(repo_id)
        if not repo or not repo.path:
            error_msg = "Error: Could not determine repository path to make file path relative."
            logging.error(error_msg)
            embedding_html_update = gr.update(value=self._generate_embedding_html(error_msg, file_path))
            code_viewer_update = gr.update(value=error_msg, language=None, label="Error", visible=True)
            image_viewer_update = gr.update(value=None, visible=False)
            no_changes_html_update = gr.update(visible=False)
            return embedding_html_update, code_viewer_update, image_viewer_update, file_path, no_changes_html_update

        try:
            repo_root = Path(repo.path).resolve()
            absolute_file_path = Path(file_path).resolve()
            relative_file_path = str(absolute_file_path.relative_to(repo_root))
            # Convert to forward slashes for consistency, as git typically uses them
            relative_file_path = relative_file_path.replace("\\", "/")
            logging.info(f"Converted absolute path '{absolute_file_path}' to relative path '{relative_file_path}' for repo '{repo_root}'")
        except ValueError as e:
            error_msg = f"Error: Could not make path relative: {e}. Absolute: {file_path}, Repo Root: {repo_root}"
            logging.error(error_msg)
            embedding_html_update = gr.update(value=self._generate_embedding_html(error_msg, file_path))
            code_viewer_update = gr.update(value=error_msg, language=None, label="Error", visible=True)
            image_viewer_update = gr.update(value=None, visible=False)
            no_changes_html_update = gr.update(visible=False)
            return embedding_html_update, code_viewer_update, image_viewer_update, file_path, no_changes_html_update

        new_selected_file_for_viewers = relative_file_path # This state should hold the relative path

        # --- Embedding Tab Update ---
        # The _generate_embedding_html function now produces HTML with an embedded script.
        # We need to inject repo_id and branch_name into that script.
        # The `raw_content_for_embedding` is no longer directly passed to _generate_embedding_html for display.

        is_image = relative_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))

        if not is_image:
            # Ensure branch_name and relative_file_path are URL-encoded for the iframe src
            import urllib.parse
            encoded_branch_name = urllib.parse.quote_plus(branch)
            encoded_relative_file_path = urllib.parse.quote_plus(relative_file_path)

            iframe_url = f"/static/ast_visualization.html?repo_id={str(repo_id)}&branch_name={encoded_branch_name}&file_path={encoded_relative_file_path}"
            embedding_html_content = f'<iframe src="{iframe_url}" style="width:100%; height:580px; border:none;"></iframe>'
            embedding_html_update = gr.update(value=embedding_html_content)
            logging.info(f"[handle_file_explorer_select] Set iframe for AST viewer: {iframe_url}")
        else:
            embedding_html_update = gr.update(value=f"<div style='padding:10px;'>AST visualization is not available for image: {relative_file_path}</div>")
            logging.info(f"[handle_file_explorer_select] AST visualization skipped for image: {relative_file_path}")
        # --- End Embedding Tab Update ---

        # --- Raw Diff Tab Update ---
        code_viewer_update = gr.skip()
        image_viewer_update = gr.skip()
        no_changes_html_update = gr.skip()

        if is_image:
            # For images, try to load the image directly for the Raw Diff tab
            # This reuses part of the logic from get_file_diff_or_content for images
            full_abs_path_for_image = repo_root / relative_file_path
            try:
                img_obj = Image.open(full_abs_path_for_image)
                img_obj.load() # Ensure image data is loaded
                image_viewer_update = gr.update(value=img_obj, visible=True)
                code_viewer_update = gr.update(visible=False)
                no_changes_html_update = gr.update(visible=False)
            except Exception as e:
                logging.error(f"Error opening image {full_abs_path_for_image} for Raw Diff tab: {e}")
                image_viewer_update = gr.update(value=None, visible=False)
                code_viewer_update = gr.update(value=f"// Error opening image: {e}", language=None, label="Error", visible=True)
                no_changes_html_update = gr.update(visible=False)
        else:
            # For non-images, determine if modified, new, or clean
            status = self.framework.get_repository_status(repo_id)
            file_status = "clean" # Default to clean
            if status:
                if relative_file_path in status.get('modified', []):
                    file_status = "modified"
                elif relative_file_path in status.get('new', []):
                    file_status = "new"
                # Add other statuses if needed (e.g., deleted, renamed)

            diff_content_for_tab = None
            lang = IngestionConfig.LANGUAGE_MAPPING.get(Path(relative_file_path).suffix.lower())

            if file_status == "modified":
                # Get actual diff
                diff_content_for_tab, _ = self.framework.get_file_diff_or_content(repo_id, relative_file_path, is_new_file_from_explorer=False)
                if diff_content_for_tab is None: # Should not happen if modified, but as a fallback
                    diff_content_for_tab = f"// Error: Could not get diff for modified file {relative_file_path}"
                code_viewer_update = gr.update(value=diff_content_for_tab, language=lang or 'diff', label=f"Diff: {relative_file_path}", visible=True)
                image_viewer_update = gr.update(visible=False)
                no_changes_html_update = gr.update(visible=False)
            elif file_status == "new":
                # Get raw content for new files
                diff_content_for_tab = self.framework.get_file_content_by_path(repo_id, branch, relative_file_path)
                if diff_content_for_tab is None or diff_content_for_tab.startswith("Error: File"): # Check for framework error string
                    diff_content_for_tab = f"// Error: Could not load content for new file {relative_file_path}. DB Error: {diff_content_for_tab}"
                code_viewer_update = gr.update(value=diff_content_for_tab, language=lang, label=f"Content (New File): {relative_file_path}", visible=True)
                image_viewer_update = gr.update(visible=False)
                no_changes_html_update = gr.update(visible=False)
            elif file_status == "clean":
                # Display "NO CHANGES"
                code_viewer_update = gr.update(value="", visible=False) # Clear and hide code viewer
                image_viewer_update = gr.update(visible=False)
                no_changes_html_update = gr.update(visible=True)
            else: # Fallback for unknown status or if status is None
                diff_content_for_tab = self.framework.get_file_content_by_path(repo_id, branch, relative_file_path)
                if diff_content_for_tab is None or diff_content_for_tab.startswith("Error: File"):
                     diff_content_for_tab = f"// Error: Could not load content for file {relative_file_path} (unknown status). DB Error: {diff_content_for_tab}"
                code_viewer_update = gr.update(value=diff_content_for_tab, language=lang, label=f"Content: {relative_file_path}", visible=True)
                image_viewer_update = gr.update(visible=False)
                no_changes_html_update = gr.update(visible=False)
        # --- End Raw Diff Tab Update ---

        # Diagnostic logging for the HTML content being sent to embedding_html_viewer
        final_html_for_viewer = "ERROR: HTML content was not correctly set"
        is_skip_update_val = False

        if isinstance(embedding_html_update, dict) and embedding_html_update.get("__type__") == "skip_update":
            is_skip_update_val = True
            final_html_for_viewer = "GR.SKIP_UPDATE"
        elif isinstance(embedding_html_update, dict) and embedding_html_update.get("__type__") == "update" and 'value' in embedding_html_update:
            final_html_for_viewer = embedding_html_update['value']
        elif isinstance(embedding_html_update, str): # Should ideally not happen if gr.update() is used
             final_html_for_viewer = embedding_html_update

        logging.info(f"[handle_file_explorer_select] Attempting to update embedding_html_viewer. Is Skip: {is_skip_update_val}. Final HTML (preview): {str(final_html_for_viewer)[:500]}...")

        return (
            embedding_html_update,
            code_viewer_update,
            image_viewer_update,
            new_selected_file_for_viewers,
            no_changes_html_update
            # ast_trigger_update  # Removed
        )

    def handle_content_tab_select(self, evt: gr.SelectData, repo_id: int, branch: str, selected_file: str) -> gr.update:
        # evt.value will be the ID of the selected tab_item (e.g., "change_analysis_tab")
        # For gr.Tabs, evt.index gives the integer index, evt.value might be None or the id if provided to TabItem.
        # Let's assume we check against the id="change_analysis_tab". Or index 1.
        # Gradio's SelectData for Tabs usually gives `index` (int) and `value` (label of TabItem).

        # Check if the "Change Analysis" tab is selected. Its ID is "change_analysis_tab".
        # The gr.Tabs component itself is named `content_tabs`.
        # The TabItems are "Embedding" (id="embedding_tab"), "Change Analysis" (id="change_analysis_tab"), "Diff" (id="diff_tab").
        # Gradio's .select event on Tabs provides SelectData with evt.value being the *label* of the TabItem.

        if evt.value == "Change Analysis": # evt.value is the label of the TabItem
            logging.info(f"Change Analysis tab selected. Current file: {selected_file}")
            if repo_id and branch and selected_file:
                status = self.framework.get_repository_status(repo_id)
                modified_files = []
                if status:
                    modified_files.extend(status.get('modified', []))
                    modified_files.extend(status.get('new', []))

                if selected_file in modified_files:
                    logging.info(f"File {selected_file} is modified. Triggering analysis.")
                    # Call the existing handler for analyzing current changes.
                    # This handler uses gr.Progress, which should work here too.
                    return self.handle_analyze_current_file_changes(repo_id, branch, selected_file)
                else:
                    logging.info(f"File {selected_file} is not in modified list for Change Analysis tab auto-trigger.")
                    return gr.update(value="Selected file is not modified. Use controls below to analyze specific changes or compare versions.")
            else:
                return gr.update(value="Select a repository, branch, and file to enable automatic change analysis.")
        return gr.skip() # No update for other tabs or if conditions not met


    def update_all_panels(self, repo_id: int, branch: str) -> Tuple:
        stats_upd, lang_upd = self.update_insights_dashboard(repo_id, branch)

        file_explorer_upd = gr.skip() # Default to skipping update for file explorer
        if repo_id and branch:
            repo = self.framework.get_repository_by_id(repo_id)
            if repo and Path(repo.path).is_dir():
                # Ensure the correct branch is materialized by get_file_tree (which does checkout)
                # get_file_tree also conveniently returns a list of files, but we ignore it here
                # as FileExplorer will read from the filesystem.
                self.framework.get_file_tree(repo_id, branch) # This call ensures checkout
                file_explorer_upd = gr.update(root_dir=str(repo.path), glob="**/*") # Update root to selected repo
            else:
                # No valid repo selected, or repo path doesn't exist
                from sda.config import WORKSPACE_DIR as SDA_WORKSPACE_DIR
                file_explorer_upd = gr.update(root_dir=str(SDA_WORKSPACE_DIR), glob=None) # Show empty or base workspace
        else:
            # No repo/branch selected, reset FileExplorer to base workspace view or empty
            from sda.config import WORKSPACE_DIR as SDA_WORKSPACE_DIR
            file_explorer_upd = gr.update(root_dir=str(SDA_WORKSPACE_DIR), value=[], glob=None) # Show empty by default

        initial_code_view_text = "// Select a file from the explorer."
        code_view_upd = gr.update(value=initial_code_view_text, language=None, label="File Content / Diff", visible=True)
        img_view_upd = gr.update(value=None, visible=False)
        sel_file_upd = ""

        embedding_html_upd = gr.update(value="<div>Select a non-image file for embedding visualization.</div>")
        change_analysis_output_upd = gr.update(value="Select an analysis option.")

        # Corrected: handle_populate_change_analysis_inputs now returns only one value
        branch_compare_older_version_ca_upd = self.handle_populate_change_analysis_inputs(repo_id, branch)

        # Removed call to obsolete self.update_git_status_panel and its result unpacking
        # mod_files_dd_diff_upd, code_v_upd_from_git, img_v_upd_from_git, sel_file_s_upd_from_git = \
        #     self.update_git_status_panel(repo_id)

        # Default updates for these, as they are no longer set by update_git_status_panel
        final_code_view_upd = code_view_upd # Use the default from earlier in the function
        final_img_view_upd = img_view_upd   # Use the default
        final_sel_file_upd = sel_file_upd   # Use the default
        no_changes_html_initial_upd = gr.update(visible=False) # Default for no_changes_message_html

        # commit_msg_diff_upd = gr.update(value="") # This component was removed

        # Order must match all_insight_outputs + doc_comp_outputs
        # all_insight_outputs = [stats_cards_html, lang_plot] (2)
        # doc_comp_outputs = [ # Now 8 items
        #     file_explorer,
        #     embedding_html_viewer,
        #     change_analysis_output,
        #     code_viewer,
        #     image_viewer,
        #     selected_file_state,
        #     branch_compare_older_version_ca,
        #     no_changes_message_html
        # ] Total 2 + 8 = 10 outputs.

        return (stats_upd, lang_upd,  # Insight outputs (2)
                file_explorer_upd,   # Doc Comp (1/8)
                embedding_html_upd, change_analysis_output_upd,  # Doc Comp (3/8)
                final_code_view_upd, final_img_view_upd, final_sel_file_upd, # Doc Comp (6/8)
                branch_compare_older_version_ca_upd, # Doc Comp (7/8)
                no_changes_html_initial_upd) # Doc Comp (8/8)
                # Total 2 + 8 = 10 outputs.

    def _generate_embedding_html(self, file_content: str, file_path_for_display: str) -> str: # Renamed arg
        logging.info(f"[_generate_embedding_html] Called for file: {file_path_for_display}. Content length: {len(file_content if file_content else '')}") # file_content is not directly used anymore for main view

        # These error conditions are now primarily handled by the API endpoint.
        # The JS will call the API. If the API returns an error, JS should display it.
        # If file_path_for_display is empty, JS won't be called with valid params.

        # Get current repo_id and branch_name from states if possible (this is tricky from a simple HTML generation function)
        # For now, assume these will be available to the JS via Gradio's mechanisms or passed explicitly.
        # The call to this function is from handle_file_explorer_select, which has repo_id and branch.
        # We need to pass these to the JS.

        # Let's reconstruct how repo_id and branch are available:
        # handle_file_explorer_select(self, repo_id: int, branch: str, selection_event_data: Any)
        # It gets repo_id and branch. It also determines relative_file_path.
        # This information needs to be embedded into the script call.

        # The `file_content` parameter to `_generate_embedding_html` is no longer the primary source of data for display.
        # It was previously used for the snippet. We will rely on the API call.
        # The `file_path_for_display` (which is the relative_file_path) is crucial for the API call.

        if not file_path_for_display:
            return "<div style='padding:10px; border:1px solid #ccc; height: 580px; overflow-y:auto;'><h2>AST Visualization</h2><p>No file selected. Please select a file from the explorer.</p></div>"

        # The repo_id and branch_name are not directly available in _generate_embedding_html's signature.
        # This function is called by handle_file_explorer_select, which *does* have repo_id and branch.
        # So, handle_file_explorer_select needs to pass these to _generate_embedding_html,
        # or _generate_embedding_html needs to be refactored to not need them directly for the HTML structure,
        # and the JS call needs to be constructed with these values in handle_file_explorer_select.

        # Let's assume `handle_file_explorer_select` will construct the script call with the correct parameters.
        # `_generate_embedding_html` will now just return the container and the script structure.
        # The actual API parameters will be filled in by `handle_file_explorer_select`.

        # The `file_path_for_display` here is the relative path.
        # api_call_url = f"/api/repo/{{repo_id}}/branch/{{branch_name}}/file-ast?path={file_path_for_display}" # Placeholders for repo/branch

        # script_content = f"""
        # <div id="ast-visualization-container" style="height: 580px; overflow-y: auto; font-family: monospace; padding: 10px; border: 1px solid #ccc;">
        #     Loading AST for {file_path_for_display}...
        # </div>
        # <script>
        # console.log('[AST Viz] SCRIPT TAG EXECUTION STARTED');
        # (async function() {{
        #     // These values will be replaced by Gradio when rendering the HTML update
        #     const repoId = "{{repo_id}}";
        #     const branchName = "{{branch_name}}";
        #     const filePath = "{file_path_for_display}"; // Already correctly escaped by f-string for JS string literal

        #     console.log("[AST Viz] Initializing script for file:", filePath);
        #     console.log("[AST Viz] Raw injected repoId:", repoId, "branchName:", branchName);

        #     const container = document.getElementById('ast-visualization-container');
        #     if (!repoId || repoId === "None" || repoId === "null" || !branchName || branchName === "None" || branchName === "null" || !filePath) {{
        #         let errorMsg = "<p>Error: Missing or invalid data to load AST:<ul>";
        #         if (!repoId || repoId === "None" || repoId === "null") errorMsg += "<li>Repository ID missing</li>";
        #         if (!branchName || branchName === "None" || branchName === "null") errorMsg += "<li>Branch name missing</li>";
        #         if (!filePath) errorMsg += "<li>File path missing</li>";
        #         errorMsg += "</ul></p>";
        #         console.error("[AST Viz] Validation failed:", errorMsg);
        #         container.innerHTML = errorMsg;
        #         return;
        #     }}

        #     // Ensure repoId is treated as a number if it's a numeric string, though API path uses it as string.
        #     // The main concern is it not being "None" or "null" as a string.
        #     console.log(`[AST Viz] Validated params - Repo ID: ${{repoId}}, Branch: ${{(branchName)}}, File: ${{filePath}}`);

        #     const apiUrl = `/api/repo/${{repoId}}/branch/${{encodeURIComponent(branchName)}}/file-ast?path=${{encodeURIComponent(filePath)}}`;
        #     console.log("[AST Viz] Fetching AST data from:", apiUrl);

        #     try {{
        #         const response = await fetch(apiUrl);
        #         if (!response.ok) {{
        #             const errorData = await response.json();
        #             throw new Error(`API Error (${{response.status}}): ${{errorData.detail || response.statusText}}`);
        #         }}
        #         const astData = await response.json();

        #         container.innerHTML = ''; // Clear loading message
        #         if (astData.length === 0) {{
        #             container.innerHTML = "<p>No AST data found for this file (it might be empty, a non-code file, or not parsed).</p>";
        #             return;
        #         }}
        #         renderAST(astData, container);
        #     }} catch (error) {{
        #         console.error("Failed to load or render AST:", error);
        #         container.innerHTML = `<p style='color:red;'>Failed to load AST data: ${{error.message}}</p>`;
        #     }}

        #     function renderAST(nodes, parentElement) {{
        #         nodes.forEach(node => {{
        #             const nodeDiv = document.createElement('div');
        #             nodeDiv.className = 'ast-node';
        #             nodeDiv.style.marginLeft = (node.depth * 15) + 'px'; // Indentation based on depth
        #             nodeDiv.style.border = '1px dashed #ddd';
        #             nodeDiv.style.padding = '5px';
        #             nodeDiv.style.marginBottom = '5px';
        #             nodeDiv.style.whiteSpace = 'pre-wrap'; // Preserve whitespace and newlines in code

        #             // Store metadata in data attributes for tooltip
        #             nodeDiv.dataset.nodeType = node.type;
        #             nodeDiv.dataset.nodeName = node.name || 'N/A';
        #             nodeDiv.dataset.startLine = node.start_line;
        #             nodeDiv.dataset.endLine = node.end_line;
        #             // Add more data attributes as needed (degree, connectivity, tokens)

        #             // Simple header for the node
        #             const header = document.createElement('div');
        #             header.className = 'ast-node-header';
        #             header.textContent = `[${{node.type}}] ${{node.name || ''}} (L${{node.start_line}}-L${{node.end_line}})`;
        #             header.style.fontSize = '0.9em';
        #             header.style.color = '#666';
        #             header.style.marginBottom = '3px';
        #             nodeDiv.appendChild(header);

        #             const codeContent = document.createElement('div');
        #             codeContent.className = 'ast-node-code';
        #             codeContent.textContent = node.code_snippet;
        #             nodeDiv.appendChild(codeContent);

        #             // Tooltip setup
        #             const tooltip = document.createElement('div');
        #             tooltip.className = 'ast-tooltip';
        #             tooltip.style.position = 'absolute';
        #             tooltip.style.visibility = 'hidden';
        #             tooltip.style.backgroundColor = 'black';
        #             tooltip.style.color = 'white';
        #             tooltip.style.padding = '5px';
        #             tooltip.style.borderRadius = '3px';
        #             tooltip.style.zIndex = '1000';
        #             tooltip.style.fontSize = '0.8em';
        #             document.body.appendChild(tooltip); // Append to body to avoid clipping issues

        #             nodeDiv.addEventListener('mouseover', (e) => {{
        #                 nodeDiv.style.backgroundColor = '#f0f0f0';
        #                 tooltip.innerHTML = `Type: ${{nodeDiv.dataset.nodeType}}<br>Name: ${{nodeDiv.dataset.nodeName}}<br>Lines: ${{nodeDiv.dataset.startLine}}-${{nodeDiv.dataset.endLine}}`;
        #                 tooltip.style.visibility = 'visible';
        #             }});
        #             nodeDiv.addEventListener('mousemove', (e) => {{
        #                 tooltip.style.left = (e.pageX + 10) + 'px';
        #                 tooltip.style.top = (e.pageY + 10) + 'px';
        #             }});
        #             nodeDiv.addEventListener('mouseout', () => {{
        #                 nodeDiv.style.backgroundColor = '';
        #                 tooltip.style.visibility = 'hidden';
        #             }});

        #             parentElement.appendChild(nodeDiv);

        #             if (node.children && node.children.length > 0) {{
        #                 renderAST(node.children, nodeDiv); // Recursive call for children, append to current nodeDiv
        #             }}
        #         }});
        #     }}
        # }})();
        # </script>
        # """
        # # The actual repoId and branchName will be injected by the calling Python function (handle_file_explorer_select)
        # # This is a placeholder structure for the script.
        # # The key is that `handle_file_explorer_select` must format this string.
        # # logging.info(f"[_generate_embedding_html] Returning HTML with script for AST visualization for {file_path_for_display}.")
        # return script_content # This will be further processed by handle_file_explorer_select

        # This function now returns an HTML template string.
        # Placeholders {repo_id}, {branch_name}, {file_path} will be replaced by the caller.
        # The file_content argument is now entirely vestigial.

        logging.info(f"[_generate_embedding_html] Generating HTML structure for AST viewer for: {file_path_for_display}")

        if not file_path_for_display: # Should be caught by caller, but as a safeguard
            return "<div id='ast-visualization-container' style='padding:10px;'>Error: No file path provided to _generate_embedding_html.</div>"

        html_template = f"""
<div id="ast-visualization-container"
     data-repo-id="{{repo_id}}"
     data-branch-name="{{branch_name}}"
     data-file-path="{{file_path}}"
     style="height: 580px; overflow-y: auto; font-family: monospace; padding: 10px; border: 1px solid #ccc;">
    Loading AST for {{file_path_display}}...
</div>
<script>
    // Brief delay to ensure the div is in the DOM if Gradio updates asynchronously
    setTimeout(() => {{
        if (typeof window.loadAndRenderAST === 'function') {{
            console.log('[AST Viz Initializer] Calling window.loadAndRenderAST()');
            window.loadAndRenderAST();
        }} else {{
            console.error('[AST Viz Initializer] window.loadAndRenderAST is not defined. Ensure ast_viewer.js is loaded correctly.');
            const container = document.getElementById('ast-visualization-container');
            if (container) {{
                container.innerHTML = "<p style='color:red;'>Error: AST viewer script (ast_viewer.js) not loaded or function not defined.</p>";
            }}
        }}
    }}, 100); // 100ms delay, adjust if needed
</script>
"""
        # Note: The placeholders {{repo_id}}, {{branch_name}}, {{file_path}}, {{file_path_display}}
        # will be replaced by handle_file_explorer_select.
        # file_path_for_display is used here just for the initial log.
        return html_template

    def handle_populate_change_analysis_inputs(self, repo_id: int, branch: str) -> gr.update:
        # Returns update for: branch_compare_older_version_ca
        if not repo_id or not branch:
            return gr.update(choices=[], value=None)

        branches = self.framework.get_repository_branches(repo_id)
        # Default older version branch to current branch, user can change it
        branches_update = gr.update(choices=branches, value=branch)

        return branches_update

    def handle_analyze_current_file_changes(self, repo_id: int, branch: str, file_path: str, progress=gr.Progress(track_tqdm=True)) -> gr.update:
        if not file_path:
            return gr.update(value="Please select a modified file to analyze.")

        progress(0, desc="Fetching diff for current changes...")
        # In a real scenario:
        # 1. Get the diff of file_path (e.g., using self.framework.get_file_diff_or_content)
        # diff_text, _ = self.framework.get_file_diff_or_content(repo_id, file_path)
        # 2. Send diff_text to an LLM with appropriate prompts.
        # llm_response = self.framework.llm_analyze_diff(diff_text)
        # For now, simulate:
        diff_text, _ = self.framework.get_file_diff_or_content(repo_id, file_path, is_new_file_from_explorer=False) # Get diff
        if diff_text is None: diff_text = "No textual changes found or file is binary."

        progress(0.5, desc="LLM analyzing changes (simulated)...")
        llm_response = f"### Analysis of Current Changes for `{file_path}` (Branch: `{branch}`):\n\n"
        llm_response += "This is a **simulated** LLM analysis.\n\n"
        llm_response += f"**Technical Assessment:**\n- The changes in `{file_path}` appear to be [simulated assessment: minor refactoring / significant feature addition / bug fix].\n"
        llm_response += "- Potential impact on other modules: [simulated: Low / Medium / High].\n"
        llm_response += "- Code quality: [simulated: Good, follows conventions / Needs improvement in XYZ areas].\n\n"
        llm_response += "**Summary of Changes (based on diff):**\n```diff\n"
        llm_response += diff_text[:1000] + ("..." if len(diff_text) > 1000 else "") # Show a snippet of the diff
        llm_response += "\n```\n"
        progress(1, desc="Analysis complete.")
        return gr.update(value=llm_response)

    def handle_analyze_version_comparison(self, repo_id: int, current_branch: str, file_to_compare: str,
                                          older_branch: str, older_version: str,
                                          progress=gr.Progress(track_tqdm=True)) -> gr.update:
        if not file_to_compare or not older_branch or not older_version or not current_branch:
            return gr.update(value="Please ensure a file is selected from the File Explorer, and specify the branch and version/commit for its older state.")

        # Current version is implicitly HEAD of current_branch for file_to_compare
        current_version_desc = f"{current_branch} @ HEAD"
        older_version_desc = f"{older_branch} @ {older_version}"

        progress(0, desc=f"Fetching versions of {Path(file_to_compare).name} for comparison...")

        # Placeholder for actual framework calls to get file content at specific versions
        # content_current = self.framework.get_file_content_at_version(repo_id, file_to_compare, current_branch, "HEAD")
        # content_older = self.framework.get_file_content_at_version(repo_id, file_to_compare, older_branch, older_version)
        # diff = generate_diff(content_older, content_current)
        # llm_analysis_of_diff = self.framework.llm_analyze_diff(diff, context=f"Comparison of {file_to_compare}")

        simulated_diff_snippet = f"--- a/{file_to_compare} ({older_version_desc})\n+++ b/{file_to_compare} ({current_version_desc})\n@@ -1,3 +1,3 @@\n- old line 1\n- old line 2\n+ new line 1\n+ new line 2\n  common line 3"

        progress(0.5, desc="LLM analyzing differences (simulated)...")
        llm_response = f"### Analysis of Differences for `{file_to_compare}`\n\n"
        llm_response += f"Comparing **{older_version_desc}** (Older) vs. **{current_version_desc}** (Current)\n\n"
        llm_response += "This is a **simulated** LLM analysis.\n\n"
        llm_response += "**Key Simulated Differences Noted:**\n- Function `example_func` was modified.\n- Class `ImportantClass` had a method signature change.\n\n"
        llm_response += "**Simulated Diff Snippet:**\n```diff\n"
        llm_response += simulated_diff_snippet + "\n```\n"
        progress(1, desc="Analysis complete.")
        return gr.update(value=llm_response)


    def handle_analyze_branch(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repository and a branch first.",) + self._get_task_button_updates(True)
        self.framework.analyze_branch(repo_id, branch, 'user')
        return (f"Started re-analysis for branch '{branch}'.",) + self._get_task_button_updates(False)

    def update_insights_dashboard(self, repo_id: int, branch: str) -> Tuple[Optional[str], Optional[px.pie]]: # Return HTML string now
        if not repo_id or not branch:
            return None, None

        stats = self.framework.get_repository_stats(repo_id, branch)

        stats_html = "" # Initialize HTML string
        if not stats:
            stats_html = "<div class='stat-cards-container'><p>No statistics available for this repository/branch.</p></div>"
        else:
            # Data for the stat cards
            file_count = stats.get('file_count', 0)
            total_lines = stats.get('total_lines', 0)
            total_tokens = stats.get('total_tokens', 0)
            schema_count = stats.get('schema_count', 0)

            # HTML structure for stat cards
            # Using f-strings for simplicity; for more complex HTML, a template engine might be better
            # but Gradio's HTML component takes a string.
            stats_html = f"""
            <div class="stat-cards-container">
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-file-alt"></i></div>
                    <div class="stat-value">{file_count:,}</div>
                    <div class="stat-label">FILES</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-stream"></i></div>
                    <div class="stat-value">{total_lines:,}</div>
                    <div class="stat-label">TOTAL LINES</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-cubes"></i></div>
                    <div class="stat-value">{schema_count}</div>
                    <div class="stat-label">SUB-MODULES</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-brain"></i></div>
                    <div class="stat-value">{total_tokens:,}</div>
                    <div class="stat-label">TOTAL TOKENS</div>
                </div>
            </div>
            """
            # The :, adds thousand separators to the numbers if they are large.

        # Prepare data for Language Breakdown Pie Chart (remains the same)
        lang_breakdown = stats.get('language_breakdown', {}) if stats else {}
        lang_fig = None
        if lang_breakdown:
            # Custom capitalization for specific languages, fallback to .capitalize()
            lang_name_map = {
                "javascript": "JavaScript", "typescript": "TypeScript",
                "html": "HTML", "css": "CSS", "json": "JSON", "xml": "XML",
                "markdown": "Markdown", "python": "Python", "java": "Java",
                "csharp": "C#", "cpp": "C++", "c": "C", "go": "Go", "rust": "Rust",
                "ruby": "Ruby", "php": "PHP", "swift": "Swift", "kotlin": "Kotlin",
                "scala": "Scala", "shell": "Shell", "powershell": "PowerShell"
                # Add more as needed
            }

            capitalized_lang_names = [lang_name_map.get(lang.lower(), lang.capitalize()) for lang in lang_breakdown.keys()]
            lang_counts = list(lang_breakdown.values())

            # Create a DataFrame for Plotly
            lang_df_for_plot = pd.DataFrame({"Language": capitalized_lang_names, "Files": lang_counts})
            lang_df_for_plot = lang_df_for_plot.sort_values(by="Files", ascending=False)

            lang_fig = px.pie(lang_df_for_plot, names="Language", values="Files",
                              hole=0.3) # Optional: for a donut chart effect, title removed

            # Styling hover labels
            lang_fig.update_traces(
                textinfo='none', # Removes text from slices
                hoverinfo='label+percent+value', # Show Language, Percentage, and File Count on hover
                hoverlabel=dict(
                    bgcolor="rgba(50,50,50,0.8)", # Semi-transparent dark background for hover
                    font_color="white",            # White font for hover text
                    font_size=12
                )
            )
            lang_fig.update_layout(
                showlegend=True,
                legend_title_text='Languages' # Optional: add a title to the legend
            )

        return stats_html, lang_fig

    def handle_run_dead_code(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repo and branch.",) + self._get_task_button_updates(True)
        self.framework.find_dead_code_for_repo(repo_id, branch, 'user')
        return ("Task to find unused code started.",) + self._get_task_button_updates(False)

    def handle_run_duplicates(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repo and branch.",) + self._get_task_button_updates(True)
        self.framework.find_duplicate_code_for_repo(repo_id, branch, 'user')
        return ("Task to find duplicate code started.",) + self._get_task_button_updates(False)

    # update_git_status_panel, handle_view_diff, handle_revert_file, handle_commit are removed
    # as their UI components have been removed from the "Raw Diff" (formerly "Diff") tab.

    def handle_code_item_select(self, repo_id: int, branch: str, df: pd.DataFrame, evt: gr.SelectData) -> gr.update:
        if evt.value is None or not isinstance(evt.index, (list, tuple)) or len(evt.index) == 0:
            return gr.update()
        try:
            row_data = df.iloc[evt.index[0]]
            selected_file = row_data.get("File", row_data.get("File A"))
        except (TypeError, AttributeError, KeyError, IndexError):
            return gr.update(value="// Could not determine file path from selection.")
        content = self.framework.get_file_content_by_path(repo_id, branch, selected_file)
        lang = IngestionConfig.LANGUAGE_MAPPING.get(Path(selected_file).suffix)
        return gr.update(value=content, language=lang, label=f"Content of: {selected_file}")


if __name__ == "__main__":
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    # Create dummy files if they don't exist to prevent Gradio errors on launch
    user_avatar_path = assets_dir / "user.png"
    bot_avatar_path = assets_dir / "bot.png"
    if not user_avatar_path.exists():
        Image.new('RGB', (100, 100), color = 'blue').save(user_avatar_path)
    if not bot_avatar_path.exists():
        Image.new('RGB', (100, 100), color = 'green').save(bot_avatar_path)

    # Initialize the framework and UI
    framework_instance = CodeAnalysisFramework()
    dashboard = DashboardUI(framework_instance)
    gradio_ui_blocks = dashboard.create_ui() # This is the gr.Blocks instance

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan_manager(app_ref: FastAPI): # Renamed parameter to avoid confusion
        # Code to run on startup
        global main_event_loop
        try:
            main_event_loop = asyncio.get_running_loop()
            print("Lifespan startup: Main event loop captured.")
        except RuntimeError:
            print("Lifespan startup: No running event loop found by get_running_loop().")
            main_event_loop = None

        yield
        # Code to run on shutdown (if any)
        print("Lifespan shutdown.")

    # Create a FastAPI app with the lifespan manager
    app = FastAPI(lifespan=lifespan_manager)

    # Add the WebSocket route
    app.add_api_websocket_route("/ws/controlpanel", websocket_control_panel_endpoint)

    # --- API Endpoint for Task History ---
    @app.get("/api/repositories/{repo_id}/tasks_history", response_model=List[TaskRead])
    async def api_get_task_history(
        repo_id: int,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=50)
    ):
        tasks = framework_instance.get_task_history(repo_id=repo_id, offset=offset, limit=limit)
        return tasks
    # --- End API Endpoint ---

    # --- API Endpoint for AST Data ---
    from sda.core.models import ASTNode as SQLA_ASTNode, File as SQLA_File # Use SQLA_ prefix for SQLAlchemy models
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional # Ensure these are imported

    class ASTVisualizationNode(BaseModel):
        id: str # node_id
        type: str
        name: Optional[str] = None
        start_line: int
        end_line: int
        start_column: Optional[int] = None
        end_column: Optional[int] = None
        depth: int
        token_count: Optional[int] = None
        dgraph_degree: Optional[str] = 'N/A' # Placeholder
        children_count: int = 0
        code_snippet: str
        children: List['ASTVisualizationNode'] = [] # Recursive definition

    # This is needed for Pydantic to handle recursive models
    ASTVisualizationNode.model_rebuild()


    def build_ast_hierarchy(
        nodes_data: List[Dict[str, Any]], # List of dicts from SQLAlchemy objects
        source_lines: List[str],
        parent_id: Optional[str] = None
    ) -> List[ASTVisualizationNode]:
        hierarchy = []
        for node_dict in nodes_data:
            if node_dict['parent_id'] == parent_id:
                # Extract code snippet for the current node
                # ASTNode line numbers are 1-based, list indices are 0-based
                snippet_lines = source_lines[node_dict['start_line'] - 1 : node_dict['end_line']]
                code_snippet = "".join(snippet_lines)

                # Calculate token count (simple space-based split for now)
                token_count = len(code_snippet.split())

                children = build_ast_hierarchy(nodes_data, source_lines, node_dict['node_id'])

                vis_node = ASTVisualizationNode(
                    id=node_dict['node_id'],
                    type=node_dict['node_type'],
                    name=node_dict['name'],
                    start_line=node_dict['start_line'],
                    end_line=node_dict['end_line'],
                    start_column=node_dict.get('start_column'), # Use .get for safety
                    end_column=node_dict.get('end_column'),   # Use .get for safety
                    depth=node_dict['depth'],
                    token_count=token_count,
                    dgraph_degree='N/A', # Placeholder
                    children_count=len(children),
                    code_snippet=code_snippet,
                    children=children
                )
                hierarchy.append(vis_node)
        # Sort by start_line to maintain order
        hierarchy.sort(key=lambda n: n.start_line)
        return hierarchy

    @app.get("/api/repo/{repo_id}/branch/{branch_name:path}/file-ast", response_model=List[ASTVisualizationNode])
    async def get_file_ast_visualization(
        repo_id: int,
        branch_name: str, # FastAPI will receive the decoded path here
        path: str = Query(..., description="Relative path to the file, POSIX style (e.g., 'src/main.py')")
    ):
        framework: CodeAnalysisFramework = framework_instance # Access the global framework instance

        repo = framework.get_repository_by_id(repo_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
        if not repo.db_schemas:
            raise HTTPException(status_code=404, detail="Repository has not been analyzed or has no data schemas.")

        # Normalize the input path to ensure it's POSIX style (though Query description already suggests it)
        normalized_relative_path = Path(path).as_posix()

        file_record: Optional[SQLA_File] = None
        ast_nodes_sqla: List[SQLA_ASTNode] = []

        for schema_name in repo.db_schemas:
            with framework.db_manager.get_session(schema_name) as session:
                # Query for the file
                current_file_record = session.query(SQLA_File).filter(
                    SQLA_File.repository_id == repo_id,
                    SQLA_File.branch == branch_name,
                    SQLA_File.relative_path == normalized_relative_path
                ).options(joinedload(SQLA_File.ast_nodes)).first() # Eager load ast_nodes

                if current_file_record:
                    file_record = current_file_record
                    # ASTNodes are already loaded due to joinedload
                    ast_nodes_sqla = sorted(file_record.ast_nodes, key=lambda n: (n.start_line, n.start_column or 0))
                    break # Found the file and its nodes

        if not file_record:
            raise HTTPException(status_code=404, detail=f"File '{normalized_relative_path}' not found in branch '{branch_name}' for this repository.")

        if not ast_nodes_sqla:
            # File exists but no AST nodes, return empty list for visualization (could be non-code file or parsing issue)
            return []

        # Read file content
        try:
            full_file_path = Path(file_record.file_path)
            if not full_file_path.is_file():
                raise HTTPException(status_code=500, detail=f"File path '{file_record.file_path}' from database does not exist on disk.")
            source_code = full_file_path.read_text(encoding='utf-8')
            source_lines = source_code.splitlines(keepends=True)
        except Exception as e:
            logging.error(f"Error reading file content for AST viz: {file_record.file_path}, Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not read file content: {e}")

        # Convert SQLAlchemy ASTNode objects to dictionaries for easier processing
        # This is important because SQLAlchemy objects might have session state issues
        # when passed around, especially if the session closes.
        ast_nodes_data = []
        for node_sqla in ast_nodes_sqla:
            ast_nodes_data.append({
                "node_id": node_sqla.node_id,
                "node_type": node_sqla.node_type,
                "name": node_sqla.name,
                "start_line": node_sqla.start_line,
                "end_line": node_sqla.end_line,
                "start_column": node_sqla.start_column,
                "end_column": node_sqla.end_column,
                "parent_id": node_sqla.parent_id, # Crucial for hierarchy building
                "depth": node_sqla.depth,
            })

        # Build hierarchical structure
        # The parent_id for top-level nodes in the file should be None or a file-level sentinel.
        # Assuming top-level nodes have parent_id = None or a specific file-level ID not matching any node_id.
        # The ASTNode model stores parent_id which can be None for root nodes of the file.
        hierarchical_ast = build_ast_hierarchy(ast_nodes_data, source_lines, parent_id=None)

        return hierarchical_ast
    # --- End API Endpoint for AST Data ---

    # Mount static files
    static_files_path = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_files_path), name="static")

    # Mount the Gradio app
    app = gr.mount_gradio_app(app, gradio_ui_blocks, path="/gradio")

    print("FastAPI app with Gradio and WebSocket endpoint is ready.")
    print(f"Access Gradio UI at http://127.0.0.1:7860/gradio (or your configured host/port)")
    print(f"Control Panel WebSocket will be at ws://127.0.0.1:7860/ws/controlpanel")

    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)