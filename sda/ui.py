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
from datetime import datetime, timezone # For time elapsed
try:
    import torch
except ImportError:
    torch = None # type: ignore

import gradio as gr
import pandas as pd
from PIL import Image
from llama_index.core.llms import ChatMessage

from app import CodeAnalysisFramework
from sda.core.models import Task # Added Task import
from sda.config import IngestionConfig, AIConfig, PG_DB_NAME, DGRAPH_HOST, DGRAPH_PORT


class DashboardUI:
    """Manages the Gradio UI components and their interactions."""

    def __init__(self, framework: CodeAnalysisFramework):
        self.framework = framework
        self.task_buttons: List[gr.Button] = []

    def _get_task_button_updates(self, interactive: bool) -> Tuple[gr.update, ...]:
        """Helper to generate a tuple of updates for all task-related buttons."""
        return tuple(gr.update(interactive=interactive) for _ in self.task_buttons)

    def _extract_progress_from_html(self, html_content: str) -> float:
        """Extract progress percentage from HTML content for comparison."""
        try:
            import re
            match = re.search(r'data-content-hash="([^"]*)"', html_content)
            if match:
                content_hash = match.group(1)
                progress_str = content_hash.split('_')[0]
                return float(progress_str)
        except:
            pass
        return 0.0

    def _create_html_progress_bar(self, progress: float, message: str = "", task_name: str = "") -> str:
        """Creates an HTML progress bar with CSS styling and JavaScript animations."""
        progress = max(0, min(100, progress))
        
        # Create a unique hash for the content to help with comparison
        content_hash = f"{progress:.1f}_{message}_{task_name}"
        
        return f"""
        <div class="progress-container" data-content-hash="{content_hash}">
            <div class="progress-header">
                <span class="progress-task-name">{task_name}</span>
                <span class="progress-percentage">{progress:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {progress}%"></div>
            </div>
            <div class="progress-message">{message}</div>
        </div>
        """

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
</div>"""
        
        html_parts = []
        html_parts.append(f"""
        <div class="status-container">
            <div class='model-info-container section-card'>
                <h4><i class="fas fa-brain"></i> Model Information</h4>
                <p><i class="fas fa-comments"></i> <strong>LLM:</strong> {AIConfig.ACTIVE_LLM_MODEL}</p>
                <p><i class="fas fa-project-diagram"></i> <strong>Embedding Model:</strong> {AIConfig.ACTIVE_EMBEDDING_MODEL}</p>
                <p><i class="fas fa-cogs"></i> <strong>Embedding Devices:</strong> {AIConfig.EMBEDDING_DEVICES}</p>
            </div>
            {self._get_hardware_info_html()}
            {self._get_storage_info_html()}
            {self._get_usage_stats_html()}
            <div class="main-task-section-card section-card">
                <h3><i class="fas fa-tasks"></i> Main Task: {task.name}</h3>
                <div class="task-status-card"> <!-- Keep this for specific task content styling -->
                    <div class="status-header">
                        <span class="status-label">Status:</span>
                    <span class="status-value">{task.status}</span>
                </div>
                {self._create_html_progress_bar(task.progress, task.message, task.name)}
                    <div class="task-timing-info">
                        {self._get_task_timing_html(task)}
                    </div>
                </div> <!-- End of task-status-card -->
            </div> <!-- End of main-task-section-card -->
        """)
        
        if task.details:
            html_parts.append("<div class='task-details section-card'><strong>Details:</strong><ul>")
            for k, v in sorted(task.details.items()):
                html_parts.append(f"<li><strong>{k}:</strong> {v}</li>")
            html_parts.append("</ul></div>")
        
        if task.children:
            html_parts.append("<h4>Sub-Tasks</h4>")
            for child in sorted(task.children, key=lambda t: t.started_at):
                if child.status == 'running':
                    status_icon = '<i class="fas fa-sync fa-spin"></i>'
                elif child.status == 'completed':
                    status_icon = '<i class="fas fa-check-circle" style="color: green;"></i>'
                else: # Error or other states
                    status_icon = '<i class="fas fa-times-circle" style="color: red;"></i>'
                html_parts.append(f"""
                <div class="subtask-card">
                    <div class="subtask-header">
                        <span class="status-icon">{status_icon}</span>
                        <span class="subtask-name">{child.name}</span>
                        <span class="subtask-status">({child.status})</span>
                    </div>
                    {self._create_html_progress_bar(child.progress, child.message, child.name)}
                    """)
                
                if child.details:
                    html_parts.append("<div class='subtask-details'><strong>Details:</strong><ul>")
                    for k, v in sorted(child.details.items()):
                        html_parts.append(f"<li><strong>{k}:</strong> {v}</li>")
                    html_parts.append("</ul></div>")
                
                html_parts.append("</div>")
        
        if task.error_message:
            error_icon = '<i class="fas fa-exclamation-triangle" style="color: orange;"></i>'
            html_parts.append(f"""
            <div class="error-section">
                <h4>{error_icon} Error</h4>
                <pre class="error-message">{task.error_message}</pre>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)

    def _get_task_timing_html(self, task) -> str:
        """Generates HTML for displaying task timing information (elapsed/remaining)."""
        if not task or not hasattr(task, 'started_at') or task.started_at is None:
            return ""

        # Ensure started_at is a timezone-aware datetime object
        started_at = task.started_at
        if isinstance(started_at, (int, float)): # Handle timestamps
            started_at = datetime.fromtimestamp(started_at, timezone.utc)
        elif hasattr(started_at, 'tzinfo') and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        timing_html = ""

        if task.status in ['running', 'pending']:
            now = datetime.now(timezone.utc)
            elapsed_seconds = (now - started_at).total_seconds()
            if elapsed_seconds < 0: elapsed_seconds = 0

            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_str = ""
            if hours > 0: elapsed_str += f"{int(hours)}h "
            if minutes > 0 or hours > 0: elapsed_str += f"{int(minutes)}m "
            elapsed_str += f"{int(seconds)}s"
            timing_html = f"<p><i class='far fa-clock'></i> <strong>Time Elapsed:</strong> {elapsed_str}</p>"

            # Placeholder for Estimated Time Remaining (ETR)
            # ETR calculation would require task.progress and a stable progress rate.
            # For now, we'll just show elapsed.
            # if task.status == 'running' and task.progress > 0 and task.progress < 100:
            #     try:
            #         # This is a very basic ETR, assumes linear progress
            #         total_estimated_time = elapsed_seconds / (task.progress / 100)
            #         remaining_seconds = total_estimated_time - elapsed_seconds
            #         if remaining_seconds > 0:
            #             r_hours, r_remainder = divmod(remaining_seconds, 3600)
            #             r_minutes, r_seconds = divmod(r_remainder, 60)
            #             etr_str = ""
            #             if r_hours > 0: etr_str += f"{int(r_hours)}h "
            #             if r_minutes > 0 or r_hours > 0: etr_str += f"{int(r_minutes)}m "
            #             etr_str += f"{int(r_seconds)}s"
            #             timing_html += f"<p><i class='fas fa-hourglass-half'></i> <strong>Est. Time Remaining:</strong> {etr_str}</p>"
            #     except ZeroDivisionError:
            #         pass # Progress is 0, cannot estimate

        elif task.status in ['completed', 'failed']:
            if hasattr(task, 'completed_at') and task.completed_at is not None:
                completed_at = task.completed_at
                if isinstance(completed_at, (int, float)): # Handle timestamps
                    completed_at = datetime.fromtimestamp(completed_at, timezone.utc)
                elif hasattr(completed_at, 'tzinfo') and completed_at.tzinfo is None:
                    completed_at = completed_at.replace(tzinfo=timezone.utc)

                if started_at and completed_at:
                    total_duration_seconds = (completed_at - started_at).total_seconds()
                    if total_duration_seconds < 0: total_duration_seconds = 0 # Should not happen

                    hours, remainder = divmod(total_duration_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_str = ""
                    if hours > 0: duration_str += f"{int(hours)}h "
                    if minutes > 0 or hours > 0: duration_str += f"{int(minutes)}m "
                    duration_str += f"{int(seconds)}s"
                    timing_html = f"<p><i class='fas fa-stopwatch'></i> <strong>Total Duration:</strong> {duration_str}</p>"
                else:
                    timing_html = "<p><i class='fas fa-stopwatch'></i> <strong>Total Duration:</strong> N/A</p>"
            else:
                timing_html = "<p><i class='fas fa-stopwatch'></i> <strong>Total Duration:</strong> N/A</p>"

        return timing_html

    def _get_hardware_info_html(self) -> str:
        # For now, we'll just show elapsed.
        # if task.status == 'running' and task.progress > 0 and task.progress < 100:
        #     try:
        #         # This is a very basic ETR, assumes linear progress
        #         total_estimated_time = elapsed_seconds / (task.progress / 100)
        #         remaining_seconds = total_estimated_time - elapsed_seconds
        #         if remaining_seconds > 0:
        #             r_hours, r_remainder = divmod(remaining_seconds, 3600)
        #             r_minutes, r_seconds = divmod(r_remainder, 60)
        #             etr_str = ""
        #             if r_hours > 0: etr_str += f"{int(r_hours)}h "
        #             if r_minutes > 0 or r_hours > 0: etr_str += f"{int(r_minutes)}m "
        #             etr_str += f"{int(r_seconds)}s"
        #             timing_html += f"<p><strong>Est. Time Remaining:</strong> {etr_str}</p>"
        #     except ZeroDivisionError:
        #         pass # Progress is 0, cannot estimate

        return timing_html

    def _get_hardware_info_html(self) -> str:
        """Generates HTML for displaying hardware and worker information."""
        num_cpus = os.cpu_count()

        allowed_db_workers = sum(IngestionConfig.MAX_DB_WORKERS_PER_TARGET.values())
        allowed_embedding_workers = AIConfig.MAX_EMBEDDING_WORKERS
        total_allowed_workers = allowed_db_workers + allowed_embedding_workers

        # GPU Info
        gpu_info_parts = []
        if torch and torch.cuda.is_available():
            gpu_info_parts.append(f"<p><strong>CUDA Version:</strong> {torch.version.cuda}</p>")
            num_gpus = torch.cuda.device_count()
            gpu_info_parts.append(f"<p><strong>Available GPUs:</strong> {num_gpus}</p>")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info_parts.append(f"<p>  - GPU {i}: {gpu_name}</p>")
        else:
            gpu_info_parts.append("<p><strong>GPU:</strong> CUDA not available or PyTorch not installed with CUDA support.</p>")

        gpu_html = "".join(gpu_info_parts)

        # Worker details
        worker_details_parts = ["<p><strong>Worker Pool Configuration:</strong></p><ul>"]
        for target, num_workers in IngestionConfig.MAX_DB_WORKERS_PER_TARGET.items():
            worker_details_parts.append(f"<li>{target.capitalize()} Workers: {num_workers}</li>")
        worker_details_parts.append(f"<li>Embedding Workers: {AIConfig.MAX_EMBEDDING_WORKERS}</li>")
        worker_details_parts.append("</ul>")
        worker_html = "".join(worker_details_parts)

        # System Load
        cpu_load = psutil.cpu_percent(interval=None) # Non-blocking, gets the last measurement
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_used_gb = ram.used / (1024**3)
        ram_percent_used = ram.percent

        return f"""
        <div class='hardware-info-container section-card'>
            <h4><i class="fas fa-server"></i> System & Worker Information</h4>
            <p><i class="fas fa-microchip"></i> <strong>Logical CPUs:</strong> {num_cpus}</p>
            <p><i class="fas fa-tachometer-alt"></i> <strong>Current CPU Load:</strong> {cpu_load:.1f}%</p>
            <p><i class="fas fa-memory"></i> <strong>RAM Usage:</strong> {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB ({ram_percent_used:.1f}%)</p>
            {gpu_html}
            <p><i class="fas fa-users-cog"></i> <strong>Total Allowed Application Workers:</strong> {total_allowed_workers}</p>
            {worker_html}
        </div>
        """

    def _get_storage_info_html(self) -> str:
        """Generates HTML for displaying storage usage information."""
        # Postgres
        pg_size_bytes = self.framework.get_postgres_db_size()
        pg_size_str = "N/A"
        if pg_size_bytes is not None:
            if pg_size_bytes < 1024:
                pg_size_str = f"{pg_size_bytes} Bytes"
            elif pg_size_bytes < 1024**2:
                pg_size_str = f"{pg_size_bytes/1024:.2f} KB"
            elif pg_size_bytes < 1024**3:
                pg_size_str = f"{pg_size_bytes/1024**2:.2f} MB"
            else:
                pg_size_str = f"{pg_size_bytes/1024**3:.2f} GB"

        # Dgraph (currently placeholder)
        dgraph_usage_str = self.framework.get_dgraph_disk_usage()
        if dgraph_usage_str is None:
            dgraph_usage_str = "N/A"

        return f"""
        <div class='storage-info-container section-card'>
            <h4><i class="fas fa-database"></i> Storage Usage</h4>
            <p><i class="fas fa-hdd"></i> <strong>Postgres ({PG_DB_NAME}):</strong> {pg_size_str}</p>
            <p><i class="fas fa-project-diagram"></i> <strong>Dgraph ({DGRAPH_HOST}:{DGRAPH_PORT}):</strong> {dgraph_usage_str}</p>
        </div>
        """

    def _get_usage_stats_html(self) -> str:
        """Generates HTML for displaying usage statistics."""
        stats = self.framework.get_usage_statistics()

        general_stats_html = f"""
            <p><i class="fas fa-folder-open"></i> <strong>Repositories Managed:</strong> {stats['general']['num_repositories']}</p>
        """
        # Add more general stats here if they get implemented in the backend, e.g.:
        # <p><i class="far fa-file-alt"></i> <strong>Total Files Analyzed:</strong> {stats['general']['total_files_analyzed']:,}</p>
        # <p><i class="fas fa-stream"></i> <strong>Total Lines Analyzed:</strong> {stats['general']['total_lines_analyzed']:,}</p>


        ai_models_html_parts = ["<ul>"]
        if stats['ai']['models_used']:
            for model, data in stats['ai']['models_used'].items():
                ai_models_html_parts.append(f"<li><strong>{model}:</strong> {data['calls']:,} calls, {data['tokens']:,} tokens, ${data['cost']:.4f}</li>")
        else:
            ai_models_html_parts.append("<li>No AI model usage tracked yet.</li>")
        ai_models_html_parts.append("</ul>")
        ai_models_html = "".join(ai_models_html_parts)

        ai_stats_html = f"""
            <p><i class="fas fa-robot"></i> <strong>Total LLM Calls:</strong> {stats['ai']['total_llm_calls']:,}</p>
            <p><i class="fas fa-brain"></i> <strong>Total Tokens Processed (LLM):</strong> {stats['ai']['total_tokens_processed']:,}</p>
            <p><i class="fas fa-dollar-sign"></i> <strong>Estimated LLM Cost:</strong> ${stats['ai']['estimated_cost']:.4f}</p>
            <div><strong>Model Breakdown:</strong>{ai_models_html}</div>
        """

        return f"""
        <div class='usage-stats-container section-card'>
            <h4><i class="fas fa-chart-line"></i> Usage Statistics</h4>
            <div class="usage-section">
                <h5>General</h5>
                {general_stats_html}
            </div>
            <div class="usage-section">
                <h5>AI Usage (LLM)</h5>
                {ai_stats_html}
            </div>
        </div>
        """

    def _format_task_log_html(self, tasks: List[Task], existing_html: str = "") -> str:
        """Formats a list of tasks into an HTML string for the log."""
        if not tasks and not existing_html: # No tasks ever and nothing existing
            return "<div class='task-log-entry'>No task history found.</div>"
        if not tasks and existing_html: # No new tasks, return existing
             return existing_html

        html_parts = [existing_html] if existing_html else []

        for task in tasks:
            status_icon_map = {
                'running': '<i class="fas fa-sync fa-spin" style="color: #007bff;"></i>',
                'pending': '<i class="fas fa-hourglass-start" style="color: #ffc107;"></i>',
                'completed': '<i class="fas fa-check-circle" style="color: green;"></i>',
                'failed': '<i class="fas fa-times-circle" style="color: red;"></i>',
            }
            status_icon = status_icon_map.get(task.status, '<i class="fas fa-question-circle"></i>')

            started_at_str = task.started_at.strftime('%Y-%m-%d %H:%M:%S UTC') if task.started_at else "N/A"
            completed_at_str = task.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC') if task.completed_at else "N/A"
            duration_str = ""
            if task.started_at and task.completed_at:
                duration_seconds = (task.completed_at - task.started_at).total_seconds()
                if duration_seconds < 0: duration_seconds = 0
                h, rem = divmod(duration_seconds, 3600)
                m, s = divmod(rem, 60)
                duration_str = f" ({int(h)}h {int(m)}m {int(s)}s)" if h > 0 else f" ({int(m)}m {int(s)}s)"

            entry = f"""
            <div class="task-log-entry section-card">
                <div class="task-log-header">
                    <span class="task-log-status-icon">{status_icon}</span>
                    <span class="task-log-name">{task.name} (ID: {task.id})</span>
                    <span class="task-log-status">Status: {task.status}</span>
                </div>
                <div class="task-log-body">
                    <p><strong>Created by:</strong> {task.created_by}</p>
                    <p><strong>Started:</strong> {started_at_str}</p>
                    """
            if task.status in ['completed', 'failed']:
                entry += f"<p><strong>Completed:</strong> {completed_at_str}{duration_str}</p>"

            entry += f"<p><strong>Message:</strong> {task.message or 'N/A'}</p>"

            if task.details:
                details_str = ", ".join([f"<strong>{k}:</strong> {v}" for k,v in task.details.items()])
                entry += f"<p><strong>Details:</strong> {details_str}</p>"

            if task.error_message:
                entry += f"<div class='task-log-error'><strong>Error:</strong> <pre>{task.error_message}</pre></div>"

            if task.log_history: # The detailed log from the task itself
                 entry += f"""
                    <details class="task-log-history-details">
                        <summary>View Raw Log Output</summary>
                        <pre class="task-log-raw-output">{task.log_history}</pre>
                    </details>
                 """
            entry += "</div></div>"
            html_parts.append(entry)

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

        modal_css = """
        .modal-background { 
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background-color: rgba(0,0,0,0.6); display: none; 
            justify-content: center; align-items: center; z-index: 1000; 
        }
        .modal-content-wrapper { 
            background-color: var(--panel-background-fill); 
            color: var(--body-text-color); padding: 2rem; 
            border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            height: auto; 
        }
        .model-info-container {{
            background: var(--background-fill-secondary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color-primary);
        }}
        .model-info-container h4, .hardware-info-container h4, .storage-info-container h4 {{
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--body-text-color);
        }}
        .model-info-container p, .hardware-info-container p, .storage-info-container p {{
            margin: 5px 0;
            font-size: 0.9em;
            color: var(--body-text-color-subdued);
        }}
        .model-info-container i, .hardware-info-container i, .storage-info-container i, .usage-stats-container i {{
            margin-right: 8px;
            color: var(--accent-color-primary); /* Or a specific color */
        }}
        .usage-stats-container h4 i {{ /* Icon in the main title of usage stats */
             color: var(--body-text-color); /* Or keep accent if preferred */
        }}
        .usage-stats-container .usage-section {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color-primary-muted); /* Muted border for inner sections */
        }}
        .usage-stats-container .usage-section:first-of-type {{
            margin-top: 0;
            padding-top: 0;
            border-top: none;
        }}
        .usage-stats-container h5 {{
            margin-bottom: 8px;
            color: var(--body-text-color);
            font-size: 1em;
            font-weight: 600;
        }}
        .usage-stats-container ul {{
            list-style-type: none;
            padding-left: 10px;
            margin-top: 5px;
        }}
        .usage-stats-container li {{
            font-size: 0.85em;
            margin-bottom: 3px;
            color: var(--body-text-color-subdued);
        }}
        .section-card {{
            background: var(--background-fill-secondary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color-primary);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .hardware-info-container h4 {{
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--body-text-color);
        }}
        .hardware-info-container p {{
            margin: 5px 0;
            font-size: 0.9em;
            color: var(--body-text-color-subdued);
        }}
        .hardware-info-container ul {{
            padding-left: 20px;
            margin: 5px 0;
        }}
        .hardware-info-container li {{
            font-size: 0.9em;
            color: var(--body-text-color-subdued);
        }}
        .task-timing-info {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color-primary);
        }}
        .task-timing-info p {{
            margin: 4px 0;
            font-size: 0.85em;
            color: var(--body-text-color-subdued);
        }}
        /* Task Log Styles */
        .task-log-entry {
            margin-bottom: 15px;
            padding: 12px;
        }
        .task-log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color-primary-muted);
        }
        .task-log-status-icon {
            margin-right: 10px;
            font-size: 1.1em;
        }
        .task-log-name {
            font-weight: 600;
            color: var(--body-text-color);
            flex-grow: 1;
        }
        .task-log-status {
            font-size: 0.9em;
            color: var(--body-text-color-subdued);
        }
        .task-log-body p {
            margin: 4px 0;
            font-size: 0.9em;
            color: var(--body-text-color-subdued);
        }
        .task-log-body strong {
            color: var(--body-text-color);
        }
        .task-log-error {
            margin-top: 8px;
            padding: 8px;
            background-color: rgba(220, 53, 69, 0.05); /* Light red background */
            border: 1px solid rgba(220, 53, 69, 0.2);
            border-radius: 4px;
        }
        .task-log-error pre {
            white-space: pre-wrap;
            word-break: break-all;
            font-size: 0.85em;
            max-height: 150px;
            overflow-y: auto;
        }
        .task-log-history-details {
            margin-top: 8px;
        }
        .task-log-history-details summary {
            cursor: pointer;
            font-size: 0.9em;
            color: var(--link-text-color);
            margin-bottom: 5px;
        }
        .task-log-raw-output {
            background-color: var(--code-background-fill);
            padding: 8px;
            border-radius: 4px;
            font-size: 0.8em;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-color-primary);
        }
        #addRepoModal .modal-content-wrapper { max-width: 500px; width: 100%; }
        #codeViewerModal .modal-content-wrapper { 
            max-width: 80vw; width: 100%; max-height: 80vh; overflow-y: auto; 
        }
        #statusModal .modal-content-wrapper { 
            max-width: 70vw; max-height: 80vh; overflow-y: auto; width: 100%; 
        }
        .model-info { 
            font-size: 0.8rem; color: var(--body-text-color-subdued); 
        }
        
        /* Prevent flicker on HTML component updates */
        .progress-wrapper {
            transition: none !important;
            animation: none !important;
        }
        
        /* Progress Bar Styles */
        .progress-container {
            width: 100%; margin: 10px 0; padding: 8px; 
            background: var(--background-fill-secondary); 
            border-radius: 8px; border: 1px solid var(--border-color-primary);
            transition: none !important;
        }
        .progress-header {
            display: flex; justify-content: space-between; 
            align-items: center; margin-bottom: 8px;
        }
        .progress-task-name {
            font-weight: 600; color: var(--body-text-color);
        }
        .progress-percentage {
            font-weight: 500; color: var(--body-text-color-subdued);
        }
        .progress-bar-bg {
            width: 100%; height: 20px; background-color: var(--background-fill-primary); 
            border-radius: 10px; overflow: hidden; position: relative;
        }
        .progress-bar-fill {
            height: 100%; background: linear-gradient(90deg, #007bff, #0056b3); 
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        .progress-bar-fill::after {
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        .progress-message {
            margin-top: 8px; font-size: 0.85rem; 
            color: var(--body-text-color-subdued); font-style: italic;
        }
        
        /* Status Modal Styles */
        .status-container {
            font-family: var(--font-sans); line-height: 1.5;
        }
        .main-task-section-card h3 {
            margin-top: 0; /* Already part of .section-card h4, but good to be explicit for h3 */
            margin-bottom: 15px;
            color: var(--body-text-color);
            font-size: 1.2em; /* Slightly larger for main task title */
        }
        .main-task-section-card h3 i {
            margin-right: 8px;
            color: var(--accent-color-primary);
        }
        .task-status-card { /* This is nested inside main-task-section-card */
            background: var(--background-fill-primary); /* Keep its own slightly different bg if needed */
            border-radius: 6px; /* Inner card might have slightly smaller radius */
            padding: 15px;
            margin: 0; /* No margin as it's inside a section-card padding */
            border: 1px solid var(--border-color-secondary); /* Subtle inner border */
        }
        .subtask-card { /* Subtasks can also be section-cards or have similar styling */
            background: var(--background-fill-primary); 
            border-radius: 8px; padding: 15px; margin: 15px 0; /* Added more margin for separation */
            border: 1px solid var(--border-color-primary);
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        .status-header, .subtask-header {
            display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
        }
        .status-label {
            font-weight: 600;
        }
        .status-value {
            padding: 2px 8px; border-radius: 4px; 
            background: var(--background-fill-secondary);
        }
        .status-icon {
            font-size: 1.2em;
        }
        .subtask-name {
            font-weight: 500; flex-grow: 1;
        }
        .subtask-status {
            color: var(--body-text-color-subdued);
        }
        .task-details, .subtask-details {
            margin-top: 10px; padding: 10px; 
            background: var(--background-fill-secondary); 
            border-radius: 4px; font-size: 0.9em;
        }
        .task-details ul, .subtask-details ul {
            margin: 5px 0; padding-left: 20px;
        }
        .error-section {
            margin-top: 15px; padding: 15px; 
            background: rgba(220, 53, 69, 0.1); 
            border: 1px solid rgba(220, 53, 69, 0.3); 
            border-radius: 8px;
        }
        .error-message {
            background: var(--background-fill-primary); 
            padding: 10px; border-radius: 4px; 
            font-size: 0.85em; overflow-x: auto;
        }
        """

        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="sky"), title="SDA Framework", css=modal_css, head=modal_js + fontawesome_cdn) as demo:
            gr.Markdown("# Software Development Analytics")
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False, placeholder="Status messages will appear here...", scale=4)
                view_status_modal_btn = gr.Button("View Control Panel", scale=1)
            
            # HTML-based progress bar with elem_classes to prevent flicker
            with gr.Row(visible=False) as progress_row:
                main_progress_bar = gr.HTML(
                    value=self._create_html_progress_bar(0, "Ready", "No active task"),
                    elem_id="main-progress-bar",
                    elem_classes=["progress-wrapper"]
                )

            repo_id_state = gr.State()
            branch_state = gr.State()
            selected_file_state = gr.State()
            last_status_text_state = gr.State("")
            last_progress_html_state = gr.State("")
            task_log_offset_state = gr.State(0)
            current_task_log_html_state = gr.State("")


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


            poll_inputs = [repo_id_state, branch_state, last_status_text_state, last_progress_html_state]
            # Add branch_dropdown and branch_state to the outputs of handle_polling
            # The old task_log_output (Textbox) is removed from poll_outputs as it's replaced by full_task_log_html (HTML)
            # full_task_log_html is updated by its own handlers now, not by polling.
            poll_outputs = [
                status_output, status_details_html, # task_log_output removed
                dead_code_df, duplicate_code_df, stats_df, lang_df,
                last_status_text_state, main_progress_bar, progress_row, last_progress_html_state,
                branch_dropdown, branch_state
            ] + self.task_buttons
            # Old task_log_output also needs to be removed from handle_polling return tuple
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

    def handle_polling(self, repo_id: int, branch: str, last_status_text: str, last_progress_html: str) -> Tuple:
        # Initialize updates for components that might not change
        branch_dropdown_update = gr.update()
        branch_state_update = gr.update() # For branch_state, usually no change unless explicitly set

        if not repo_id:
            default_progress_html = self._create_html_progress_bar(0, "No repository selected", "Idle")
            # task_log_output (the Textbox) has been removed from here
            no_repo_status_html = "<div class='status-container'>No repository selected.</div>"
            no_repo_updates = (
                "No repository selected.", # status_output
                gr.update(value=no_repo_status_html) if no_repo_status_html != last_status_text else gr.update(), # status_details_html
                gr.update(), # dead_code_df
                gr.update(), # duplicate_code_df
                gr.update(), # stats_df
                gr.update(), # lang_df
                no_repo_status_html, # last_status_text_state
                gr.update(value=default_progress_html) if default_progress_html != last_progress_html else gr.update(), # main_progress_bar
                gr.update(visible=False), # progress_row
                default_progress_html, # last_progress_html_state
                branch_dropdown_update, # branch_dropdown
                branch_state_update     # branch_state
            )
            return no_repo_updates + self._get_task_button_updates(True)
        
        task = self.framework.get_latest_task(repo_id) # This gets the *latest* for the status modal, not full history
        if not task:
            default_progress_html = self._create_html_progress_bar(0, "No active tasks", "Idle")
            no_task_status_html = "<div class='status-container'>No tasks found for this repository.</div>"
            # task_log_output (the Textbox) has been removed from here
            no_task_updates = (
                "No tasks found for this repository.", # status_output
                gr.update(value=no_task_status_html) if no_task_status_html != last_status_text else gr.update(), # status_details_html
                gr.update(), # dead_code_df
                gr.update(), # duplicate_code_df
                gr.update(), # stats_df
                gr.update(), # lang_df
                no_task_status_html, # last_status_text_state
                gr.update(value=default_progress_html) if default_progress_html != last_progress_html else gr.update(), # main_progress_bar
                gr.update(visible=False), # progress_row
                default_progress_html, # last_progress_html_state
                branch_dropdown_update, # branch_dropdown
                branch_state_update     # branch_state
            )
            return no_task_updates + self._get_task_button_updates(True)

        status_msg = f"Task '{task.name}': {task.message} ({task.progress:.0f}%)"
        # log_output for the old textbox is no longer needed here.
        # The status_details_html will show the current task's own log snippet if any.
        is_running = task.status in ['running', 'pending']

        # More specific check for tasks that might alter branch state
        task_could_change_branch = task.name.startswith("Ingest Branch:") or task.name == "analyze_branch" # refine if needed

        dead_code_update, dup_code_update = gr.update(), gr.update()
        stats_update, lang_update = gr.update(), gr.update()

        status_html = self._create_status_progress_html(task)
        status_details_update = gr.update() if status_html == last_status_text else gr.update(value=status_html)
        current_progress_html = self._create_html_progress_bar(task.progress, task.message, task.name)
        
        progress_changed = (
            current_progress_html != last_progress_html and 
            (last_progress_html == "" or abs(task.progress - self._extract_progress_from_html(last_progress_html)) >= 1.0)
        )
        main_progress_update = gr.update(value=current_progress_html) if progress_changed else gr.update()
        progress_row_update = gr.update(visible=is_running)

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
                # Refresh branch information
                current_repo_branches = self.framework.get_repository_branches(repo_id)
                repo = self.framework.get_repository_by_id(repo_id)
                new_active_branch = None
                if repo:
                    new_active_branch = repo.active_branch

                # Ensure new_active_branch is valid, otherwise pick first or None
                if new_active_branch not in current_repo_branches:
                    new_active_branch = current_repo_branches[0] if current_repo_branches else None

                # Update dropdown choices and value
                branch_dropdown_update = gr.update(choices=current_repo_branches, value=new_active_branch)
                # Update the branch_state if the active branch has changed
                if branch != new_active_branch: # 'branch' is the input branch_state
                    branch_state_update = new_active_branch

                # Also refresh stats if an ingestion task completed
                stats_update, lang_update = self.update_insights_dashboard(repo_id, new_active_branch or branch)


        elif task.status == 'failed':
            status_msg = f"Task '{task.name}' Failed: Check logs for details."

        button_updates = self._get_task_button_updates(interactive=not is_running)

        return (
            status_msg, status_details_update, # log_output removed
            dead_code_update, dup_code_update, stats_update, lang_update,
            status_html, main_progress_update, progress_row_update, current_progress_html,
            branch_dropdown_update, branch_state_update
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