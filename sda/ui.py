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
import psutil # For CPU load and RAM
import re # For _extract_progress_from_html
from datetime import datetime, timezone # For time elapsed
try:
    import torch
except ImportError:
    torch = None # type: ignore

import gradio as gr
import pandas as pd
from PIL import Image
from llama_index.core.llms import ChatMessage
from jinja2 import Environment, FileSystemLoader, select_autoescape # Added Jinja2 imports

from app import CodeAnalysisFramework
from sda.core.models import Task
from sda.config import IngestionConfig, AIConfig, PG_DB_NAME, DGRAPH_HOST, DGRAPH_PORT


class DashboardUI:
    """Manages the Gradio UI components and their interactions."""

    def __init__(self, framework: CodeAnalysisFramework):
        self.framework = framework
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
            "task": task,
            "model_info_html": self._get_model_info_html(),
            "hardware_info_html": self._get_hardware_info_html(),
            "storage_info_html": self._get_storage_info_html(),
            "usage_stats_html": self._get_usage_stats_html(),
            "main_task_progress_html": self._create_html_progress_bar(task.progress, task.message, task.name),
            "task_timing_html": self._get_task_timing_html(task),
            "render_sub_task": self._render_sub_task_html # Pass method to render sub_tasks
        }
        
        template = self.jinja_env.get_template("status_container_base.html")
        return template.render(context)

    def _render_sub_task_html(self, child_task: Task) -> str:
        """Renders a single sub-task using its template."""
        progress_bar_html = self._create_html_progress_bar(child_task.progress, child_task.message, child_task.name)
        template = self.jinja_env.get_template("status_modal_parts/sub_task.html")
        return template.render(task=child_task, progress_bar_html=progress_bar_html)

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
            duration_str = ""
            if task.started_at and task.completed_at:
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

            html_parts.append(template.render(task=task, duration_str=duration_str))

        return "".join(html_parts)

    def handle_load_more_tasks(self, repo_id: Optional[int], offset: int, current_html: str, limit: int = 10) -> Tuple[str, int, gr.update]:
        """Handles loading more tasks for the task log."""
        tasks = self.framework.get_task_history(repo_id=repo_id, offset=offset, limit=limit)
        new_html = self._format_task_log_html(tasks, existing_html=current_html if offset > 0 else "")
        new_offset = offset + len(tasks)

        # Disable "Load More" button if no more tasks were fetched this round
        load_more_btn_update = gr.update(interactive=bool(tasks and len(tasks) == limit))

        return new_html, new_offset, load_more_btn_update

    def create_ui(self) -> gr.Blocks:
        """Builds the Gradio Blocks UI."""

        modal_js = """
        <script>
            function showModal(id) { document.getElementById(id).style.display = 'flex'; }
            function hideModal(id) { document.getElementById(id).style.display = 'none'; return null; }
            function setupModalEventListeners() {
                document.querySelectorAll('.modal-background').forEach(modal => {
                    modal.addEventListener('click', function(event) {
                        if (event.target === modal) { hideModal(modal.id); }
                    });
                });
            }
            window.addEventListener('load', setupModalEventListeners);
            # Add CSS for preventing flicker during updates
            window.addEventListener('DOMContentLoaded', function() {
                // Prevent flicker on progress bar updates
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


        # Load CSS from file
        css_file_path = Path(__file__).parent / "static" / "css" / "control_panel.css"
        try:
            with open(css_file_path, "r") as f:
                control_panel_css = f.read()
        except FileNotFoundError:
            control_panel_css = "/* CSS file not found. Styles will be missing. */"
            print(f"Warning: CSS file not found at {css_file_path}")


        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="sky"), title="SDA Framework", css=control_panel_css, head=tailwind_cdn + modal_js + fontawesome_cdn + dynamic_updates_js_link) as demo:
            gr.Markdown("# Software Development Analytics")
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False, placeholder="Status messages will appear here...", scale=4)
                view_status_modal_btn = gr.Button("View Control Panel", scale=1)
            
            # HTML-based progress bar with elem_classes to prevent flicker
            with gr.Row(visible=False) as progress_row:
                main_progress_bar = gr.HTML(
                    value=self._create_html_progress_bar(0, "Ready", "No active task", unique_prefix="external"),
                    elem_id="main-progress-bar", # This ID is for the overall container of the HTML content
                    elem_classes=["progress-wrapper"]
                )

            repo_id_state = gr.State()
            branch_state = gr.State()
            selected_file_state = gr.State()
            last_status_text_state = gr.State("")
            last_progress_html_state = gr.State("")
            task_log_offset_state = gr.State(0)
            current_task_log_html_state = gr.State("")
            js_update_data_output = gr.JSON(visible=False, label="JS Update Data", elem_id="js_update_data_json")
            # States for tracking Control Panel structure
            last_main_task_id_state = gr.State(None)
            last_sub_task_ids_state = gr.State([])
            last_main_task_has_details_state = gr.State(False)
            last_main_task_has_error_state = gr.State(False)


            with gr.Column(elem_id="addRepoModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    gr.Markdown("## Add New Repository")
                    repo_url_modal = gr.Textbox(label="Git Repository URL or Local Path", placeholder="https://github.com/user/repo.git or /path/to/local/repo")
                    with gr.Row():
                        add_repo_submit_btn = gr.Button("Add & Analyze", variant="primary")
                        add_repo_cancel_btn = gr.Button("Cancel")

            with gr.Column(elem_id="codeViewerModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    modal_code_viewer = gr.Code(label="File Content", language=None, interactive=False)
                    modal_close_btn = gr.Button("Close")

            with gr.Column(elem_id="statusModal", elem_classes="modal-background"):
                with gr.Column(elem_classes="modal-content-wrapper"):
                    gr.Markdown("## Control Panel")
                    status_details_html = gr.HTML(value="<div class='status-container'>No active tasks.</div>")
                    status_modal_close_btn = gr.Button("Close")

            with gr.Tabs() as tabs:
                with gr.TabItem("Repository & Agent", id=0):
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

                with gr.TabItem("Insights Dashboard", id=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown('### <i class="fas fa-chart-bar"></i> Statistics')
                            stats_df = gr.DataFrame(headers=["Metric", "Value"], col_count=(2, "fixed"), interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown('### <i class="fas fa-code"></i> Language Breakdown')
                            lang_df = gr.DataFrame(headers=["Language", "Files", "Percentage"], col_count=(3, "fixed"), interactive=False)
                    gr.Markdown('### <i class="fas fa-search-plus"></i> In-Depth Analysis (runs in background)')
                    with gr.Row():
                        analyze_dead_code_btn = gr.Button("Find Potentially Unused Code")
                        analyze_duplicates_btn = gr.Button("Find Potentially Duplicate Code")
                    with gr.Accordion("Full Task History", open=False) as task_history_accordion:
                        full_task_log_html = gr.HTML("No tasks loaded yet.")
                        load_more_tasks_btn = gr.Button("Load More Tasks")
                    with gr.Tabs():
                        with gr.TabItem("Unused Code Results"):
                            dead_code_df = gr.DataFrame(headers=["File", "Symbol", "Type", "Lines"], interactive=False, max_height=400,
                                                        datatype=["str", "str", "str", "str"])
                        with gr.TabItem("Duplicate Code Results"):
                            duplicate_code_df = gr.DataFrame(headers=["File A", "Lines A", "File B", "Lines B", "Similarity"], interactive=False,
                                                             max_height=400)

                with gr.TabItem("Code Browser & Version Control", id=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Git Status")
                            modified_files_dropdown = gr.Dropdown(label="Select Modified File to View", interactive=True)
                            revert_file_btn = gr.Button("Revert Changes for Selected File")
                            commit_message = gr.Textbox(label="Commit Message", placeholder="Enter commit message...")
                            commit_btn = gr.Button("Commit All Changes")
                        with gr.Column(scale=3):
                            code_viewer = gr.Code(label="File Content / Diff", language=None, interactive=False, visible=True)
                            image_viewer = gr.Image(label="Image Content", interactive=False, visible=False)

            self.task_buttons = [analyze_branch_btn, analyze_dead_code_btn, analyze_duplicates_btn, add_repo_submit_btn, commit_btn]
            timer = gr.Timer(2)

            all_insight_outputs = [stats_df, lang_df]
            git_panel_outputs = [modified_files_dropdown, code_viewer, image_viewer, selected_file_state]

            # Define these components early if they are needed by reference in poll_outputs
            # For this specific fix, we will pass them to handle_polling and it will return gr.update() for them.
            # The actual Gradio component instances are repo_dropdown, branch_dropdown.

            # Initial load of task log (first page) when UI loads or repo changes
            # We need a way to trigger this. For now, let's make it part of handle_repo_select
            # and handle_initial_load.

            # The task_log_output is now full_task_log_html
            demo.load(
                self.handle_initial_load,
                outputs=[repo_dropdown, branch_dropdown, repo_id_state, branch_state, chatbot, task_log_offset_state, current_task_log_html_state]
            ).then(
                self.handle_load_more_tasks, # Initial load of tasks after states are reset
                inputs=[repo_id_state, task_log_offset_state, current_task_log_html_state],
                outputs=[full_task_log_html, task_log_offset_state, load_more_tasks_btn]
            ).then( # This then block might be redundant if update_all_panels doesn't affect task log
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs
            )


            poll_inputs = [
                repo_id_state, branch_state,
                last_status_text_state, last_progress_html_state,
                # New state inputs for polling structural changes
                last_main_task_id_state, last_sub_task_ids_state,
                last_main_task_has_details_state, last_main_task_has_error_state
            ]
            poll_outputs = [
                status_output, status_details_html,
                dead_code_df, duplicate_code_df, stats_df, lang_df,
                last_status_text_state, main_progress_bar, progress_row, last_progress_html_state,
                branch_dropdown, branch_state,
                js_update_data_output,
                # New state outputs from polling
                last_main_task_id_state, last_sub_task_ids_state,
                last_main_task_has_details_state, last_main_task_has_error_state
            ] + self.task_buttons
            timer.tick(self.handle_polling, poll_inputs, poll_outputs)


            open_add_repo_modal_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'flex'; }")
            add_repo_cancel_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }")
            modal_close_btn.click(None, js="() => { const modal = document.getElementById('codeViewerModal'); if (modal) modal.style.display = 'none'; }")
            view_status_modal_btn.click(None, js="() => { const modal = document.getElementById('statusModal'); if (modal) modal.style.display = 'flex'; }")
            status_modal_close_btn.click(None, js="() => { const modal = document.getElementById('statusModal'); if (modal) modal.style.display = 'none'; }")

            add_repo_submit_btn.click(
                self.handle_add_repo, [repo_url_modal], [status_output, repo_dropdown] + self.task_buttons
            ).then(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }").then(
                self.handle_repo_select, [repo_dropdown],
                [branch_dropdown, repo_id_state, branch_state, chatbot, task_log_offset_state, current_task_log_html_state] # task_log_output removed, states added
            ).then(
                self.handle_load_more_tasks, # Reload tasks for new repo
                inputs=[repo_id_state, task_log_offset_state, current_task_log_html_state],
                outputs=[full_task_log_html, task_log_offset_state, load_more_tasks_btn]
            )

            repo_dropdown.change(
                self.handle_repo_select, [repo_dropdown],
                [branch_dropdown, repo_id_state, branch_state, chatbot, task_log_offset_state, current_task_log_html_state] # task_log_output removed, states added
            ).then(
                self.handle_load_more_tasks, # Reload tasks for new repo selection
                inputs=[repo_id_state, task_log_offset_state, current_task_log_html_state],
                outputs=[full_task_log_html, task_log_offset_state, load_more_tasks_btn]
            ).then(
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs
            )

            # Branch change should also reset and reload task log if it's repo-specific
            branch_dropdown.change(
                self.handle_branch_select, [branch_dropdown],
                [branch_state, chatbot, task_log_offset_state, current_task_log_html_state] # task_log_output removed, states added
            ).then(
                self.handle_load_more_tasks, # Reload tasks for new branch selection
                inputs=[repo_id_state, task_log_offset_state, current_task_log_html_state], # Assuming repo_id_state is still valid
                outputs=[full_task_log_html, task_log_offset_state, load_more_tasks_btn]
            ).then(
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs
            )

            # Connect the "Load More Tasks" button
            load_more_tasks_btn.click(
                self.handle_load_more_tasks,
                inputs=[repo_id_state, task_log_offset_state, current_task_log_html_state],
                outputs=[full_task_log_html, task_log_offset_state, load_more_tasks_btn]
            )

            # Optional: Reload tasks when accordion is opened (if it was previously empty due to no repo)
            # This requires knowing the accordion's open state or using its change event.
            # For now, relying on repo/branch changes.

            analyze_branch_btn.click(self.handle_analyze_branch, [repo_id_state, branch_state], [status_output, progress_row] + self.task_buttons)
            analyze_dead_code_btn.click(self.handle_run_dead_code, [repo_id_state, branch_state], [status_output, progress_row] + self.task_buttons)
            analyze_duplicates_btn.click(self.handle_run_duplicates, [repo_id_state, branch_state], [status_output, progress_row] + self.task_buttons)

            dead_code_df.select(self.handle_code_item_select, [repo_id_state, branch_state, dead_code_df], [modal_code_viewer]).then(
                None, js="() => { const modal = document.getElementById('codeViewerModal'); if (modal) modal.style.display = 'flex'; }")
            modified_files_dropdown.change(self.handle_view_diff, [repo_id_state, modified_files_dropdown], [code_viewer, image_viewer, selected_file_state])
            revert_file_btn.click(self.handle_revert_file, [repo_id_state, selected_file_state], [status_output] + git_panel_outputs)
            commit_btn.click(self.handle_commit, [repo_id_state, commit_message], [status_output, commit_message] + git_panel_outputs + self.task_buttons)

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
        last_status_text: str, last_progress_html: str,
        # New state inputs
        last_main_task_id: Optional[int],
        last_sub_task_ids: List[int],
        last_main_task_has_details: bool,
        last_main_task_has_error: bool
    ) -> Tuple:
        branch_dropdown_update = gr.update()
        branch_state_update = gr.update()

        # Initialize new state outputs
        new_last_main_task_id = last_main_task_id
        new_last_sub_task_ids = list(last_sub_task_ids) # Copy
        new_last_main_task_has_details = last_main_task_has_details
        new_last_main_task_has_error = last_main_task_has_error

        js_update_data = {
            "control_panel": {"main_task": None, "sub_tasks": [], "hardware_info": None},
            "external_progress_bar": {"task_name": "Idle", "progress": 0.0, "message": "Initializing..."}
        }

        current_cpu_load = psutil.cpu_percent(interval=None)
        current_ram_percent = psutil.virtual_memory().percent
        current_ram_used_gb = psutil.virtual_memory().used / (1024**3)
        current_ram_total_gb = psutil.virtual_memory().total / (1024**3)
        js_update_data["control_panel"]["hardware_info"] = {
            "cpu_load": current_cpu_load,
            "ram_percent": current_ram_percent,
            "ram_absolute_text": f"{current_ram_used_gb:.1f} / {current_ram_total_gb:.1f} GB"
        }

        full_html_update_needed_for_modal = False

        if not repo_id:
            js_update_data["external_progress_bar"]["message"] = "No repository selected"
            default_ext_progress_html = self._create_html_progress_bar(0, "No repository selected", "Idle", unique_prefix="external")

            if last_main_task_id is not None: # Was showing a task, now isn't
                full_html_update_needed_for_modal = True
                new_last_main_task_id = None
                new_last_sub_task_ids = []
                new_last_main_task_has_details = False
                new_last_main_task_has_error = False

            current_modal_html = self._create_status_progress_html(None)
            status_details_update = gr.update(value=current_modal_html) if full_html_update_needed_for_modal or current_modal_html != last_status_text else gr.update()
            new_last_status_text_for_state = current_modal_html

            no_repo_updates_tuple = (
                "No repository selected.", status_details_update,
                gr.update(), gr.update(), gr.update(), gr.update(),
                new_last_status_text_for_state,
                gr.update(value=default_ext_progress_html) if default_ext_progress_html != last_progress_html else gr.update(),
                gr.update(visible=False),
                default_ext_progress_html,
                branch_dropdown_update, branch_state_update,
                js_update_data,
                # New state outputs
                new_last_main_task_id, new_last_sub_task_ids, new_last_main_task_has_details, new_last_main_task_has_error
            )
            return no_repo_updates_tuple + self._get_task_button_updates(True)
        
        task = self.framework.get_latest_task(repo_id)

        if not task:
            js_update_data["external_progress_bar"]["message"] = "No active tasks"
            default_ext_progress_html = self._create_html_progress_bar(0, "No active tasks", "Idle", unique_prefix="external")

            if last_main_task_id is not None: # Was showing a task, now isn't
                full_html_update_needed_for_modal = True
                new_last_main_task_id = None
                new_last_sub_task_ids = []
                new_last_main_task_has_details = False
                new_last_main_task_has_error = False

            current_modal_html = self._create_status_progress_html(None)
            status_details_update = gr.update(value=current_modal_html) if full_html_update_needed_for_modal or current_modal_html != last_status_text else gr.update()
            new_last_status_text_for_state = current_modal_html

            no_task_updates_tuple = (
                "No tasks found for this repository.", status_details_update,
                gr.update(), gr.update(), gr.update(), gr.update(),
                new_last_status_text_for_state,
                gr.update(value=default_ext_progress_html) if default_ext_progress_html != last_progress_html else gr.update(),
                gr.update(visible=False), default_ext_progress_html,
                branch_dropdown_update, branch_state_update,
                js_update_data,
                new_last_main_task_id, new_last_sub_task_ids, new_last_main_task_has_details, new_last_main_task_has_error
            )
            return no_task_updates_tuple + self._get_task_button_updates(True)

        # --- Task exists, populate js_update_data and determine updates ---
        status_msg = f"Task '{task.name}': {task.message} ({task.progress:.0f}%)"
        is_running = task.status in ['running', 'pending']

        js_update_data["external_progress_bar"]["task_name"] = task.name if is_running else "Idle"
        js_update_data["external_progress_bar"]["progress"] = task.progress if is_running else (100.0 if task.status in ['completed', 'failed'] else 0.0)
        js_update_data["external_progress_bar"]["message"] = task.message if is_running else ("Task " + task.status)

        elapsed_str, _, duration_str = self._get_task_timing_values(task)

        main_task_status_classes_map = {
            'running': "bg-blue-200 text-blue-800 dark:bg-blue-700 dark:text-blue-200",
            'completed': "bg-green-200 text-green-800 dark:bg-green-700 dark:text-green-200",
            'failed': "bg-red-200 text-red-800 dark:bg-red-700 dark:text-red-200",
            'pending': "bg-yellow-200 text-yellow-800 dark:bg-yellow-700 dark:text-yellow-200"
        }
        main_task_status_class = main_task_status_classes_map.get(task.status, "bg-gray-200 text-gray-800 dark:bg-gray-600 dark:text-gray-200")
        main_task_status_class += " px-3 py-1 text-xs font-semibold rounded-full"

        js_update_data["control_panel"]["main_task"] = {
            "name": task.name, "status_text": task.status, "status_class": main_task_status_class,
            "progress": task.progress, "message": task.message,
            "time_elapsed": elapsed_str, "time_duration": duration_str,
            "details": task.details if task.details else {},
            "error_message": task.error_message,
        }

        current_sub_task_ids = sorted([st.id for st in task.children]) if task.children else []
        current_main_task_has_details = bool(task.details)
        current_main_task_has_error = bool(task.error_message)

        # Check for structural changes in the modal content
        if (task.id != last_main_task_id or
            set(current_sub_task_ids) != set(last_sub_task_ids) or # Order doesn't matter for set comparison, but JS might need sorted IDs
            current_main_task_has_details != last_main_task_has_details or
            current_main_task_has_error != last_main_task_has_error):
            full_html_update_needed_for_modal = True

        new_last_main_task_id = task.id
        new_last_sub_task_ids = current_sub_task_ids
        new_last_main_task_has_details = current_main_task_has_details
        new_last_main_task_has_error = current_main_task_has_error

        sub_task_status_classes_map = {
            'running': "bg-blue-100 text-blue-700 dark:bg-blue-600 dark:text-blue-100",
            'completed': "bg-green-100 text-green-700 dark:bg-green-600 dark:text-green-100",
            'failed': "bg-red-100 text-red-700 dark:bg-red-600 dark:text-red-100",
            'pending': "bg-yellow-100 text-yellow-700 dark:bg-yellow-600 dark:text-yellow-100"
        }
        default_sub_task_class_base = "bg-gray-200 text-gray-700 dark:bg-gray-600 dark:text-gray-100"
        sub_task_badge_common_classes = " text-xxs px-1.5 py-0.5 rounded-full whitespace-nowrap flex-shrink-0"

        if task.children:
            for child_task in sorted(task.children, key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc)):
                child_status_class = sub_task_status_classes_map.get(child_task.status, default_sub_task_class_base) + sub_task_badge_common_classes
                js_update_data["control_panel"]["sub_tasks"].append({
                    "id": child_task.id, "name": child_task.name, "status_text": child_task.status,
                    "status_class": child_status_class, "progress": child_task.progress,
                    "message": child_task.message, "details": child_task.details if child_task.details else {}
                })

        if full_html_update_needed_for_modal:
            current_modal_html = self._create_status_progress_html(task)
            status_details_update = gr.update(value=current_modal_html)
            new_last_status_text_for_state = current_modal_html
        else: # No structural change, only values might have changed. JS will handle.
            status_details_update = gr.update()
            new_last_status_text_for_state = last_status_text # Keep old HTML state if not re-rendering

        current_ext_progress_html = self._create_html_progress_bar(
            js_update_data["external_progress_bar"]["progress"],
            js_update_data["external_progress_bar"]["message"],
            js_update_data["external_progress_bar"]["task_name"],
            unique_prefix="external"
        )
        main_progress_update = gr.update(value=current_ext_progress_html) if current_ext_progress_html != last_progress_html else gr.update()
        new_last_progress_html_for_state = current_ext_progress_html if current_ext_progress_html != last_progress_html else last_progress_html

        progress_row_update = gr.update(visible=is_running)
        button_updates = self._get_task_button_updates(interactive=not is_running)
        dead_code_update, dup_code_update = gr.update(), gr.update()
        stats_update, lang_update = gr.update(), gr.update()

        task_could_change_branch = task.name.startswith("Ingest Branch:") or task.name == "analyze_branch"
        if task.status == 'completed':
            status_msg = f"Last task '{task.name}' completed successfully."
            if task.result:
                if task.name == "find_dead_code" and task.result.get("dead_code"):
                    dead_code_data = task.result["dead_code"]
                    df_data = [[dc.get('file_path'), dc.get('name'), dc.get('node_type'), f"{dc.get('start_line')}-{dc.get('end_line')}"] for dc in dead_code_data]
                    dead_code_update = pd.DataFrame(df_data, columns=["File", "Symbol", "Type", "Lines"])
                elif task.name == "find_duplicate_code" and task.result.get("duplicate_code"):
                    dup_data = task.result["duplicate_code"]
                    df_data = [[d.get('file_a'), d.get('lines_a'), d.get('file_b'), d.get('lines_b'), f"{d.get('similarity', 0):.2%}"] for d in dup_data]
                    dup_code_update = pd.DataFrame(df_data, columns=["File A", "Lines A", "File B", "Lines B", "Similarity"])

            if task_could_change_branch:
                current_repo_branches = self.framework.get_repository_branches(repo_id)
                repo = self.framework.get_repository_by_id(repo_id)
                new_active_branch = repo.active_branch if repo else None
                if new_active_branch and new_active_branch not in current_repo_branches:
                    new_active_branch = current_repo_branches[0] if current_repo_branches else None

                branch_dropdown_update = gr.update(choices=current_repo_branches, value=new_active_branch)
                if branch != new_active_branch : branch_state_update = new_active_branch
                stats_update, lang_update = self.update_insights_dashboard(repo_id, new_active_branch or branch)

        elif task.status == 'failed':
            status_msg = f"Task '{task.name}' Failed: Check logs for details."

        return (
            status_msg, status_details_update,
            dead_code_update, dup_code_update, stats_update, lang_update,
            new_last_status_text_for_state,
            main_progress_update, progress_row_update,
            new_last_progress_html_for_state,
            branch_dropdown_update, branch_state_update,
            js_update_data,
            # New state outputs
            new_last_main_task_id, new_last_sub_task_ids, new_last_main_task_has_details, new_last_main_task_has_error
        ) + button_updates

    def handle_initial_load(self) -> Tuple[gr.update, gr.update, Optional[int], Optional[str], List[Dict[str, str]], int, str]:
        repos = self.framework.get_all_repositories()
        repo_choices = [(f"{repo.name} ({repo.path})", repo.id) for repo in repos]
        initial_repo_id = repo_choices[0][1] if repo_choices else None
        repo_upd = gr.update(choices=repo_choices, value=initial_repo_id)

        # Outputs for handle_initial_load are:
        # repo_dropdown, branch_dropdown, repo_id_state, branch_state, chatbot,
        # task_log_offset_state (to reset for handle_load_more_tasks), current_task_log_html_state (to reset)
        if not initial_repo_id:
            return repo_upd, gr.update(choices=[], value=None), None, None, [{"role": "assistant", "content": "Welcome! Please add a repository to begin."}], 0, ""

        # Call handle_repo_select to get most of the values, then add task log resets
        branch_upd, repo_id_val, branch_val, chatbot_val, _offset_reset, _html_reset = self.handle_repo_select(initial_repo_id) # type: ignore
        return repo_upd, branch_upd, repo_id_val, branch_val, chatbot_val, 0, ""


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

    def handle_repo_select(self, repo_id: int) -> Tuple[gr.update, Optional[int], Optional[str], List[Dict[str,str]], int, str]:
        # Returns: branch_dropdown_update, repo_id_state, branch_state, chatbot_update, task_log_offset_state, current_task_log_html_state
        if not repo_id:
            return gr.update(choices=[], value=None), None, None, [{"role": "assistant", "content": "Please select a repository."}], 0, ""
        repo = self.framework.get_repository_by_id(int(repo_id))
        if not repo:
            return gr.update(choices=[], value=None), repo_id, None, [], 0, ""
        branches = self.framework.get_repository_branches(repo_id)
        active_branch = repo.active_branch if repo.active_branch in branches else (branches[0] if branches else None)
        chatbot_reset = [{"role": "assistant", "content": f"Agent ready for '{repo.name}' on branch '{active_branch}'."}]
        return gr.update(choices=branches, value=active_branch), repo_id, active_branch, chatbot_reset, 0, "" # Reset offset and HTML

    def handle_branch_select(self, branch: str) -> Tuple[str, List[Dict[str,str]], int, str]:
        # Returns: branch_state, chatbot_update, task_log_offset_state, current_task_log_html_state
        chatbot_msg = [{"role": "assistant", "content": f"Agent context switched to branch '{branch}'."}]
        return branch, chatbot_msg, 0, "" # Reset offset and HTML

    def update_all_panels(self, repo_id: int, branch: str) -> Tuple:
        stats_upd, lang_upd = self.update_insights_dashboard(repo_id, branch)
        mod_files_upd, code_view_upd, img_view_upd, sel_file_upd = self.update_git_status_panel(repo_id)
        return stats_upd, lang_upd, mod_files_upd, code_view_upd, img_view_upd, sel_file_upd

    def handle_analyze_branch(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repository and a branch first.", gr.update(visible=False)) + self._get_task_button_updates(True)
        self.framework.analyze_branch(repo_id, branch, 'user')
        return (f"Started re-analysis for branch '{branch}'.", gr.update(visible=True)) + self._get_task_button_updates(False)

    def update_insights_dashboard(self, repo_id: int, branch: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not repo_id or not branch:
            return pd.DataFrame(columns=["Metric", "Value"]), pd.DataFrame(columns=["Language", "Files", "Percentage"])
        stats = self.framework.get_repository_stats(repo_id, branch)
        file_count = stats.get('file_count', 0)
        stats_data = [
            ["File Count", f"{file_count:,}"],
            ["Total Lines", f"{stats.get('total_lines', 0):,}"],
            ["Total Tokens", f"{stats.get('total_tokens', 0):,}"],
            ["Sub-modules (schemas)", f"{stats.get('schema_count', 0):,}"]
        ]
        stats_df = pd.DataFrame(stats_data, columns=["Metric", "Value"])
        lang_data = []
        lang_breakdown = stats.get('language_breakdown', {})
        for lang, count in sorted(lang_breakdown.items(), key=lambda item: item[1], reverse=True):
            percentage = f"{(count / file_count * 100):.2f}%" if file_count > 0 else "0.00%"
            lang_data.append([lang, f"{count:,}", percentage])
        lang_df = pd.DataFrame(lang_data, columns=["Language", "Files", "Percentage"])
        return stats_df, lang_df

    def handle_run_dead_code(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repo and branch.", gr.update(visible=False)) + self._get_task_button_updates(True)
        self.framework.find_dead_code_for_repo(repo_id, branch, 'user')
        return ("Task to find unused code started.", gr.update(visible=True)) + self._get_task_button_updates(False)

    def handle_run_duplicates(self, repo_id: int, branch: str) -> Tuple:
        if not repo_id or not branch:
            return ("Please select a repo and branch.", gr.update(visible=False)) + self._get_task_button_updates(True)
        self.framework.find_duplicate_code_for_repo(repo_id, branch, 'user')
        return ("Task to find duplicate code started.", gr.update(visible=True)) + self._get_task_button_updates(False)

    def update_git_status_panel(self, repo_id: int) -> Tuple:
        if not repo_id:
            return gr.update(choices=[], value=None), "No repository selected.", gr.update(visible=False), None
        status = self.framework.get_repository_status(repo_id)
        all_changed_files = (status.get('modified', []) + status.get('new', []) + status.get('untracked', []) + status.get('renamed', [])) if status else []
        dropdown_upd = gr.update(choices=all_changed_files, value=all_changed_files[0] if all_changed_files else None)
        selected_file = all_changed_files[0] if all_changed_files else None
        if selected_file:
            diff_or_err, image = self.framework.get_file_diff_or_content(repo_id, selected_file)
            if image:
                return dropdown_upd, gr.update(visible=False), gr.update(value=image, visible=True), selected_file
            else:
                return dropdown_upd, diff_or_err or "No changes.", gr.update(visible=False), selected_file
        else:
            return dropdown_upd, "No uncommitted changes found.", gr.update(visible=False), None

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

    def handle_view_diff(self, repo_id: int, file_path: str) -> Tuple:
        if not repo_id or not file_path:
            return gr.update(visible=True, value="Please select a file."), gr.update(visible=False), None
        diff_or_err, image = self.framework.get_file_diff_or_content(repo_id, file_path)
        if image:
            return gr.update(visible=False), gr.update(value=image, visible=True), file_path
        else:
            return gr.update(value=diff_or_err or "No changes to display.", visible=True), gr.update(visible=False), file_path

    def handle_revert_file(self, repo_id: int, file_path: str) -> Tuple:
        if not repo_id or not file_path:
            return ("Please select a file to revert.",) + tuple(self.update_git_status_panel(repo_id))
        self.framework.revert_file_changes(repo_id, file_path)
        status_msg = f"Reverted {file_path}."
        return (status_msg,) + tuple(self.update_git_status_panel(repo_id))

    def handle_commit(self, repo_id: int, commit_message: str) -> Tuple:
        if not repo_id:
            return ("No repository selected.", gr.update(value=commit_message)) + tuple(self.update_git_status_panel(repo_id)) + self._get_task_button_updates(True)
        if not commit_message:
            return ("Commit message cannot be empty.", gr.update(value=commit_message)) + tuple(self.update_git_status_panel(repo_id)) + self._get_task_button_updates(True)
        repo = self.framework.get_repository_by_id(repo_id)
        if not repo:
            return ("Repository not found.", gr.update(value=commit_message)) + tuple(self.update_git_status_panel(repo_id)) + self._get_task_button_updates(True)
        self.framework.git_service.stage_all(repo.path)
        self.framework.git_service.commit(repo.path, commit_message)
        status_msg = "Commit successful."
        return (status_msg, "") + tuple(self.update_git_status_panel(repo_id)) + self._get_task_button_updates(True)


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

    framework_instance = CodeAnalysisFramework()
    dashboard = DashboardUI(framework_instance)
    ui = dashboard.create_ui()
    ui.launch()