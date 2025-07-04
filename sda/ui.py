# ui.py

"""
Implements the Gradio-based user interface for the Code Analysis Framework.

This dashboard provides tools for repository management, an overview of
analysis insights, a live code editor, and version control operations.
It interacts exclusively with the CodeAnalysisFramework facade.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Generator

import gradio as gr
import pandas as pd
from PIL import Image
from llama_index.core.llms import ChatMessage

from app import CodeAnalysisFramework
from sda.config import IngestionConfig, AIConfig


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
            return "<div class='status-container'>No active tasks.</div>"
        
        html_parts = []
        html_parts.append(f"""
        <div class="status-container">
            <h3>Main Task: {task.name}</h3>
            <div class="task-status-card">
                <div class="status-header">
                    <span class="status-label">Status:</span>
                    <span class="status-value">{task.status}</span>
                </div>
                {self._create_html_progress_bar(task.progress, task.message, task.name)}
            </div>
        """)
        
        if task.details:
            html_parts.append("<div class='task-details'><strong>Details:</strong><ul>")
            for k, v in sorted(task.details.items()):
                html_parts.append(f"<li><strong>{k}:</strong> {v}</li>")
            html_parts.append("</ul></div>")
        
        if task.children:
            html_parts.append("<h4>Sub-Tasks</h4>")
            for child in sorted(task.children, key=lambda t: t.started_at):
                status_icon = "🔄" if child.status == 'running' else ("✅" if child.status == 'completed' else "❌")
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
            html_parts.append(f"""
            <div class="error-section">
                <h4>❗ Error</h4>
                <pre class="error-message">{task.error_message}</pre>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)

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
        .task-status-card, .subtask-card {
            background: var(--background-fill-primary); 
            border-radius: 8px; padding: 15px; margin: 10px 0;
            border: 1px solid var(--border-color-primary);
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

        with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="sky"), title="SDA Framework", css=modal_css, head=modal_js) as demo:
            gr.Markdown("# Software Development Analytics")
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False, placeholder="Status messages will appear here...", scale=4)
                view_status_modal_btn = gr.Button("View System Status", scale=1)
            
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
                    gr.Markdown("## System Status")
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
                    with gr.Row():
                        gr.Markdown(f"**LLM:** `{AIConfig.ACTIVE_LLM_MODEL}`", elem_classes=["model-info"])
                        gr.Markdown(f"**Embedding:** `{AIConfig.ACTIVE_EMBEDDING_MODEL}` | **Devices:** `{AIConfig.EMBEDDING_DEVICES}`",
                                    elem_classes=["model-info"])

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
                            gr.Markdown("### 📊 Statistics")
                            stats_df = gr.DataFrame(headers=["Metric", "Value"], col_count=(2, "fixed"), interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown("### 🌐 Language Breakdown")
                            lang_df = gr.DataFrame(headers=["Language", "Files", "Percentage"], col_count=(3, "fixed"), interactive=False)
                    gr.Markdown("### 🔬 In-Depth Analysis (runs in background)")
                    with gr.Row():
                        analyze_dead_code_btn = gr.Button("Find Potentially Unused Code")
                        analyze_duplicates_btn = gr.Button("Find Potentially Duplicate Code")
                    with gr.Accordion("Detailed Task Logs", open=False):
                        task_log_output = gr.Textbox(label="Logs", interactive=False, lines=15, max_lines=30, autoscroll=True)
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

            demo.load(self.handle_initial_load, outputs=[repo_dropdown, branch_dropdown, repo_id_state, branch_state, task_log_output, chatbot]).then(
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs)

            poll_inputs = [repo_id_state, branch_state, last_status_text_state, last_progress_html_state]
            poll_outputs = [status_output, task_log_output, status_details_html, dead_code_df, duplicate_code_df, stats_df, lang_df, last_status_text_state, main_progress_bar, progress_row, last_progress_html_state] + self.task_buttons
            timer.tick(self.handle_polling, poll_inputs, poll_outputs)

            open_add_repo_modal_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'flex'; }")
            add_repo_cancel_btn.click(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }")
            modal_close_btn.click(None, js="() => { const modal = document.getElementById('codeViewerModal'); if (modal) modal.style.display = 'none'; }")
            view_status_modal_btn.click(None, js="() => { const modal = document.getElementById('statusModal'); if (modal) modal.style.display = 'flex'; }")
            status_modal_close_btn.click(None, js="() => { const modal = document.getElementById('statusModal'); if (modal) modal.style.display = 'none'; }")

            add_repo_submit_btn.click(
                self.handle_add_repo, [repo_url_modal], [status_output, repo_dropdown] + self.task_buttons
            ).then(None, js="() => { const modal = document.getElementById('addRepoModal'); if (modal) modal.style.display = 'none'; }").then(
                self.handle_repo_select, [repo_dropdown], [branch_dropdown, repo_id_state, branch_state, task_log_output, chatbot]
            )

            repo_dropdown.change(self.handle_repo_select, [repo_dropdown], [branch_dropdown, repo_id_state, branch_state, task_log_output, chatbot]).then(
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs)
            branch_dropdown.change(self.handle_branch_select, [branch_dropdown], [branch_state, task_log_output, chatbot]).then(
                self.update_all_panels, [repo_id_state, branch_state], all_insight_outputs + git_panel_outputs)

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
        if not repo_id:
            default_progress_html = self._create_html_progress_bar(0, "No repository selected", "Idle")
            no_repo_updates = (
                "No repository selected.", "", 
                "<div class='status-container'>No repository selected.</div>", 
                gr.update(), gr.update(), gr.update(), gr.update(), "", 
                gr.update(value=default_progress_html) if default_progress_html != last_progress_html else gr.update(), 
                gr.update(visible=False),
                default_progress_html
            )
            return no_repo_updates + self._get_task_button_updates(True)
        
        task = self.framework.get_latest_task(repo_id)
        if not task:
            default_progress_html = self._create_html_progress_bar(0, "No active tasks", "Idle")
            no_task_updates = (
                "No tasks found for this repository.", "", 
                "<div class='status-container'>No tasks found for this repository.</div>", 
                gr.update(), gr.update(), gr.update(), gr.update(), "", 
                gr.update(value=default_progress_html) if default_progress_html != last_progress_html else gr.update(), 
                gr.update(visible=False),
                default_progress_html
            )
            return no_task_updates + self._get_task_button_updates(True)

        status_msg = f"Task '{task.name}': {task.message} ({task.progress:.0f}%)"
        log_output = task.log_history or ""
        is_running = task.status in ['running', 'pending']
        was_ingestion_task = 'ingest' in task.name.lower()

        dead_code_update, dup_code_update = gr.update(), gr.update()
        stats_update, lang_update = gr.update(), gr.update()

        # Generate detailed HTML status
        status_html = self._create_status_progress_html(task)
        
        # Anti-flicker logic: only update if content has changed
        status_details_update = gr.update() if status_html == last_status_text else gr.update(value=status_html)

        # Main progress bar update with stronger anti-flicker
        current_progress_html = self._create_html_progress_bar(task.progress, task.message, task.name)
        
        # Only update if there's a meaningful change (not just tiny progress differences)
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
                    df_data = [[dc.get('file_path'), dc.get('name'), dc.get('node_type'), f"{dc.get('start_line')}-{dc.get('end_line')}"] for dc in
                               dead_code_data]
                    dead_code_update = pd.DataFrame(df_data, columns=["File", "Symbol", "Type", "Lines"])
                elif task.name == "find_duplicate_code" and task.result.get("duplicate_code"):
                    dup_data = task.result["duplicate_code"]
                    df_data = [[d.get('file_a'), d.get('lines_a'), d.get('file_b'), d.get('lines_b'), f"{d.get('similarity', 0):.2%}"] for d in dup_data]
                    dup_code_update = pd.DataFrame(df_data, columns=["File A", "Lines A", "File B", "Lines B", "Similarity"])
            if was_ingestion_task:
                stats_update, lang_update = self.update_insights_dashboard(repo_id, branch)
        elif task.status == 'failed':
            status_msg = f"Task '{task.name}' Failed: Check logs for details."

        button_updates = self._get_task_button_updates(interactive=not is_running)
        return (status_msg, log_output, status_details_update, dead_code_update, dup_code_update, stats_update, lang_update, status_html, main_progress_update, progress_row_update, current_progress_html) + button_updates

    def handle_initial_load(self) -> Tuple:
        repos = self.framework.get_all_repositories()
        repo_choices = [(f"{repo.name} ({repo.path})", repo.id) for repo in repos]
        initial_repo_id = repo_choices[0][1] if repo_choices else None
        repo_upd = gr.update(choices=repo_choices, value=initial_repo_id)
        if not initial_repo_id:
            return repo_upd, gr.update(choices=[], value=None), None, None, "", [{"role": "assistant", "content": "Welcome! Please add a repository to begin."}]
        branch_upd, repo_id_val, branch_val, log_val, chatbot_val = self.handle_repo_select(initial_repo_id)
        return repo_upd, branch_upd, repo_id_val, branch_val, log_val, chatbot_val

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

    def handle_repo_select(self, repo_id: int) -> Tuple:
        if not repo_id:
            return gr.update(choices=[], value=None), None, None, "", [{"role": "assistant", "content": "Please select a repository."}]
        repo = self.framework.get_repository_by_id(int(repo_id))
        if not repo:
            return gr.update(choices=[], value=None), repo_id, None, "", []
        branches = self.framework.get_repository_branches(repo_id)
        active_branch = repo.active_branch if repo.active_branch in branches else (branches[0] if branches else None)
        chatbot_reset = [{"role": "assistant", "content": f"Agent ready for '{repo.name}' on branch '{active_branch}'."}]
        return gr.update(choices=branches, value=active_branch), repo_id, active_branch, "", chatbot_reset

    def handle_branch_select(self, branch: str) -> Tuple:
        chatbot_msg = [{"role": "assistant", "content": f"Agent context switched to branch '{branch}'."}]
        return branch, "", chatbot_msg

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