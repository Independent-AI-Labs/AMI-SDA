// sda/static/js/dynamic_updates.js

// --- Dynamic Update Helper Functions ---
function updateTextContent(elementId, text, isHtml = false) {
    const element = document.getElementById(elementId);
    if (element) {
        if (isHtml) {
            if (element.innerHTML !== text) element.innerHTML = text;
        } else {
            if (element.innerText !== text) element.innerText = text;
        }
    } else {
        // console.warn(`Element not found for text update: ${elementId}`);
    }
}

function updateElementClass(elementId, newClass) {
    const element = document.getElementById(elementId);
    if (element && element.className !== newClass) {
        element.className = newClass;
    } else if (!element) {
        // console.warn(`Element not found for class update: ${elementId}`);
    }
}

function updateProgressUI(uniquePrefix, progress, message, taskName) {
    const barElement = document.getElementById(uniquePrefix + '-progress-bar');
    if (barElement) {
        barElement.style.width = (progress || 0) + '%';
    } else {
        // console.warn(`Progress bar element not found: ${uniquePrefix}-progress-bar`);
    }

    const textElement = document.getElementById(uniquePrefix + '-progress-text');
    if (textElement) {
        const currentProgress = progress !== null && progress !== undefined ? progress.toFixed(0) : '0';
        const currentMessage = message || "";
        const newText = `${currentMessage} (${currentProgress}%)`;
        if (textElement.innerText !== newText) textElement.innerText = newText;
    } else {
        // console.warn(`Progress text element not found: ${uniquePrefix}-progress-text`);
    }

    const taskNameEl = document.getElementById(uniquePrefix + '-task-name');
    if (taskNameEl && taskName) {
        if(taskNameEl.innerText !== taskName) taskNameEl.innerText = taskName;
        if(taskNameEl.title !== taskName) taskNameEl.title = taskName;
    } else if (taskNameEl && !taskName) { // Clear if no task name
        if(taskNameEl.innerText !== "") taskNameEl.innerText = "";
        if(taskNameEl.title !== "") taskNameEl.title = "";
    }
}

function updateHardwareInfo(hwInfo) {
    if (!hwInfo) return;
    const cpuLoadBar = document.getElementById('cpu-load-bar');
    if(cpuLoadBar) cpuLoadBar.style.width = (hwInfo.cpu_load || 0) + '%';
    updateTextContent('cpu-load-value', (hwInfo.cpu_load !== null ? hwInfo.cpu_load.toFixed(0) : '0') + '% Load');

    const ramLoadBar = document.getElementById('ram-usage-bar');
    if(ramLoadBar) ramLoadBar.style.width = (hwInfo.ram_percent || 0) + '%';
    updateTextContent('ram-usage-value', (hwInfo.ram_absolute_text || 'N/A'));
}

function updateMainTaskDetails(detailsData) {
    const detailsList = document.getElementById('main-task-details-list');
    const detailsCard = document.getElementById('main-task-details-card');
    if (!detailsCard || !detailsList) return;

    if (detailsData && Object.keys(detailsData).length > 0) {
        detailsCard.style.display = '';
        let listContent = '';
        const sortedDetails = Object.entries(detailsData).sort((a, b) => a[0].localeCompare(b[0]));
        for (const [key, value] of sortedDetails) {
            listContent += `<li class="text-xs text-gray-600 dark:text-gray-400"><strong class="font-medium text-gray-700 dark:text-gray-300">${key}:</strong> <span class="detail-value">${value}</span></li>`;
        }
        // Only update if content actually changes to avoid unnecessary DOM manipulation
        if (detailsList.innerHTML !== listContent) detailsList.innerHTML = listContent;
    } else {
        detailsCard.style.display = 'none';
    }
}

function updateMainTaskError(errorMessage) {
    const errorCard = document.getElementById('main-task-error-card');
    const errorMsgEl = document.getElementById('main-task-error-message-content');
    if (!errorCard || !errorMsgEl) return;

    if (errorMessage) {
        errorCard.style.display = '';
        if (errorMsgEl.innerText !== errorMessage) errorMsgEl.innerText = errorMessage;
    } else {
        errorCard.style.display = 'none';
    }
}

function updateSubTaskUI(st) {
    const prefix = `sub-task-${st.id}`;
    updateTextContent(`${prefix}-name`, st.name);
    updateElementClass(`${prefix}-status-badge`, st.status_class); // status_class includes all necessary classes
    updateTextContent(`${prefix}-status-badge`, st.status_text);
    updateProgressUI(prefix, st.progress, st.message, st.name);

    const iconEl = document.getElementById(`${prefix}-status-icon`)?.querySelector('i');
    if(iconEl){
        let newIconClass = "fas text-sm "; // Base classes for icon
        if(st.status_text === 'running') newIconClass += "fa-sync fa-spin text-blue-500";
        else if(st.status_text === 'completed') newIconClass += "fa-check-circle text-green-500";
        else if(st.status_text === 'pending') newIconClass += "fa-hourglass-start text-yellow-500";
        else newIconClass += "fa-times-circle text-red-500"; // Default for failed or other
        if(iconEl.className !== newIconClass) iconEl.className = newIconClass;
    }

    const subDetailsContainer = document.getElementById(`${prefix}-details-container`);
    const subDetailsList = document.getElementById(`${prefix}-details-list`);
    if(subDetailsContainer && subDetailsList) {
        if (st.details && Object.keys(st.details).length > 0) {
            subDetailsContainer.style.display = '';
            let subListContent = '';
            const sortedSubDetails = Object.entries(st.details).sort((a, b) => a[0].localeCompare(b[0]));
            for (const [key, value] of sortedSubDetails) {
                 // Added class to value span for potential future individual updates
                subListContent += `<li class="text-gray-500 dark:text-gray-400"><strong class="font-medium text-gray-600 dark:text-gray-300">${key}:</strong> <span class="detail-value-${st.id}-${key.replace(/[^a-zA-Z0-9-_]/g, '')}">${value}</span></li>`;
            }
            if(subDetailsList.innerHTML !== subListContent) subDetailsList.innerHTML = subListContent;
        } else {
            subDetailsContainer.style.display = 'none';
        }
    }
}


// Main function to apply updates from js_update_data
function applyJSUpdates(data) {
    if (!data) {
        // console.log("applyJSUpdates called with no data or null data");
        return;
    }
    // console.log("Applying JS Updates:", JSON.stringify(data, null, 2));

    if (data.external_progress_bar) {
        const epb = data.external_progress_bar;
        // The external progress bar's HTML is simple and fully rendered by Python via _create_html_progress_bar.
        // Python's handle_polling decides whether to send a new HTML string for it.
        // So, direct JS manipulation of external-progress-bar's parts might be redundant if Python always sends full HTML for it.
        // However, if Python sends gr.update() for main_progress_bar, then this JS is needed.
        // The Python code was changed to send gr.update() for main_progress_bar if HTML string is same.
        // So these JS updates ARE needed for the external bar.
         updateProgressUI('external', epb.progress, epb.message, epb.task_name);
    }

    if (data.control_panel) {
        const cp = data.control_panel;

        if (cp.hardware_info) {
            updateHardwareInfo(cp.hardware_info);
        }

        const mainTaskData = cp.main_task;
        if (mainTaskData) { // If there's an active main task
            updateTextContent('main-task-name-value', mainTaskData.name);
            updateElementClass('main-task-status-badge', mainTaskData.status_class);
            updateTextContent('main-task-status-badge', mainTaskData.status_text);
            updateProgressUI('main-task', mainTaskData.progress, mainTaskData.message, mainTaskData.name);
            updateTextContent('main-task-time-elapsed', mainTaskData.time_elapsed || 'N/A');
            updateTextContent('main-task-time-duration', mainTaskData.time_duration || 'N/A');
            updateMainTaskDetails(mainTaskData.details);
            updateMainTaskError(mainTaskData.error_message);
        }
        // Note: If mainTaskData is null (no active task), Python's full HTML render of status_details_html
        // (from _create_status_progress_html(None)) is expected to correctly show the "No active task" message.
        // JS doesn't need to explicitly hide main task sections if Python handles the full structural change.


        // Sub-task updates
        // This simplified version assumes Python re-renders the sub-task list if tasks are added/removed.
        // This JS part will only update values of *existing* sub-task DOM elements.
        const subTasksData = cp.sub_tasks || [];
        subTasksData.forEach(st => {
            // Check if the sub-task element exists before trying to update it
            if (document.getElementById(`sub-task-${st.id}`)) {
                updateSubTaskUI(st);
            } else {
                // console.warn(`Sub-task element not found for ID: sub-task-${st.id}. Python full render should handle this.`);
            }
        });
        // TODO: More robust sub-task handling: if len(subTasksData) != number of rendered sub-task divs,
        // it implies Python should have sent a full HTML update for status_details_html.
        // If Python *didn't* send a full update but sub-task list changed, JS would need to add/remove divs.
        // For now, we rely on Python's full update for structural changes in sub-task list.
    }
}

// Observe the hidden JSON component for changes
window.addEventListener('DOMContentLoaded', (event) => {
    const targetNode = document.getElementById('js_update_data_json');
    if (targetNode) {
        const observer = new MutationObserver(function(mutationsList, observer) {
            for(const mutation of mutationsList) {
                if (mutation.type === 'childList' || mutation.type === 'characterData' || mutation.type === 'attributes') {
                    // Check if the component itself or its content has changed
                    // The actual JSON data is usually in a <pre> tag inside the Gradio component's wrapper
                    let rawText = "";
                    const preElement = targetNode.querySelector('pre');
                    if (preElement) {
                        rawText = preElement.innerText || preElement.textContent;
                    } else {
                         // Fallback for older Gradio or if structure changes
                        rawText = targetNode.innerText || targetNode.textContent;
                    }

                    if (rawText && rawText.trim() && rawText.trim().toLowerCase() !== 'null') {
                        try {
                            const jsonData = JSON.parse(rawText);
                            applyJSUpdates(jsonData);
                        } catch (e) {
                            console.error('Error parsing JS update data:', e, "Raw text:", rawText);
                        }
                    } else {
                        // console.log("JSON data is null or empty, skipping update.");
                         applyJSUpdates(null); // Call with null to potentially clear/hide elements if needed
                    }
                    break;
                }
            }
        });
        observer.observe(targetNode, { childList: true, characterData: true, subtree: true, attributes: true });
        // console.log("JS Update Observer attached to js_update_data_json.");
    } else {
        console.warn('JS Update Data JSON element (js_update_data_json) not found on DOMContentLoaded.');
    }
});

// Ensure collapsible section JS is also present (from previous steps)
function toggleCollapsible(contentId, chevronId) {
    const contentElement = document.getElementById(contentId);
    const chevronIcon = document.getElementById(chevronId)?.querySelector('i');

    if (!contentElement || !chevronIcon) return;

    const isOpen = contentElement.dataset.isOpen === 'true';

    if (isOpen) {
        contentElement.style.maxHeight = '0px';
        contentElement.style.opacity = '0';
        contentElement.dataset.isOpen = 'false';
        chevronIcon.classList.remove('fa-chevron-up');
        chevronIcon.classList.add('fa-chevron-down');
    } else {
        contentElement.style.opacity = '1';
        // Set max-height to scrollHeight for animation
        // Ensure content is visible to calculate scrollHeight correctly if it was display:none
        const prevDisplay = contentElement.style.display;
        contentElement.style.display = 'block'; // Temporarily ensure it's block for scrollHeight
        contentElement.style.maxHeight = contentElement.scrollHeight + 'px';
        contentElement.style.display = prevDisplay; // Restore if needed, though class handles visibility

        contentElement.dataset.isOpen = 'true';
        chevronIcon.classList.remove('fa-chevron-down');
        chevronIcon.classList.add('fa-chevron-up');

        setTimeout(() => {
            if (contentElement.dataset.isOpen === 'true') {
                 // Allow content to expand further if its internal size changes
                contentElement.style.maxHeight = 'none'; // Or a very large fixed value like '500vh'
            }
        }, 350); // Should be slightly longer than CSS transition duration
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize collapsible task section
    const initialTaskContent = document.getElementById('task-activity-content');
    if (initialTaskContent) {
        initialTaskContent.dataset.isOpen = 'true';
        initialTaskContent.style.opacity = '1';
        const initialChevron = document.getElementById('task-activity-chevron')?.querySelector('i');
        if (initialChevron) {
            initialChevron.classList.remove('fa-chevron-down');
            initialChevron.classList.add('fa-chevron-up');
        }
        // Set max-height after a brief moment to allow rendering, ensuring scrollHeight is accurate
        // And only if it's meant to be open
        setTimeout(() => {
            if (initialTaskContent.dataset.isOpen === 'true') {
                 if (initialTaskContent.scrollHeight > 0) {
                    initialTaskContent.style.maxHeight = initialTaskContent.scrollHeight + 'px';
                 } else {
                    // Fallback if scrollHeight is 0 (e.g. if initially no task content)
                    // but it's supposed to be open for future content.
                    initialTaskContent.style.maxHeight = '500vh'; // Large enough
                 }
            } else { // Ensure it's visually collapsed if dataset.isOpen is false
                initialTaskContent.style.maxHeight = '0px';
                initialTaskContent.style.opacity = '0';
            }
        }, 70);
    }
});
