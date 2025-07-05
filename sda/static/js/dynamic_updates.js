// sda/static/js/dynamic_updates.js

function createProgressBarHTMLForJS(uniquePrefix, progress, message, taskName) {
    // This function mimics the structure of progress_bar.html template
    // Ensure class names and structure match for consistent styling.
    const progressText = message ? `${message} (${progress.toFixed(0)}%)` : `(${progress.toFixed(0)}%)`;
    const taskNameDisplay = taskName ? `<span id="${uniquePrefix}-task-name" class="text-xs font-semibold text-gray-700 dark:text-gray-300 truncate pr-2" title="${taskName}">${taskName}</span>` : "";

    return `
        <div class="progress-wrapper mb-1">
            ${taskName || message ? `
            <p class="text-xs text-gray-600 dark:text-gray-400 mb-0.5 flex justify-between items-center">
                ${taskNameDisplay}
                <span id="${uniquePrefix}-progress-text" class="text-xxs whitespace-nowrap">${progressText}</span>
            </p>` : `
            <p class="text-xs text-gray-600 dark:text-gray-400 mb-0.5 flex justify-end items-center">
                 <span id="${uniquePrefix}-progress-text" class="text-xxs whitespace-nowrap">${progressText}</span>
            </p>
            `}
            <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div id="${uniquePrefix}-progress-bar" class="bg-blue-500 h-2 rounded-full transition-width duration-300 ease-out" style="width: ${progress}%"></div>
            </div>
        </div>
    `;
}


// --- Dynamic Update Helper Functions ---
function updateTextContent(elementId, text, isHtml = false) {
    const element = document.getElementById(elementId);
    if (element) {
        if (isHtml) {
            if (element.innerHTML !== text) element.innerHTML = text;
        } else {
            if (element.innerText !== text) element.innerText = text;
        }
    }
}

function updateElementClass(elementId, newClass) {
    const element = document.getElementById(elementId);
    if (element && element.className !== newClass) {
        element.className = newClass;
    }
}

function updateProgressUI(uniquePrefix, progress, message, taskName) {
    const barElement = document.getElementById(uniquePrefix + '-progress-bar');
    if (barElement) barElement.style.width = (progress || 0) + '%';

    const textElement = document.getElementById(uniquePrefix + '-progress-text');
    if (textElement) {
        const currentProgress = progress !== null && progress !== undefined ? progress.toFixed(0) : '0';
        const currentMessage = message || "";
        const newText = `${currentMessage} (${currentProgress}%)`;
        if (textElement.innerText !== newText) textElement.innerText = newText;
    }

    const taskNameEl = document.getElementById(uniquePrefix + '-task-name');
    if (taskNameEl) { // Update task name if element exists
        const currentTaskName = taskName || "";
        if(taskNameEl.innerText !== currentTaskName) taskNameEl.innerText = currentTaskName;
        if(taskNameEl.title !== currentTaskName) taskNameEl.title = currentTaskName;
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

function updateOrCreateSubTaskElement(subTaskData, container) {
    const prefix = `sub-task-${subTaskData.id}`;
    let subTaskElement = document.getElementById(prefix);

    if (!subTaskElement) { // Create new sub-task element
        const template = document.getElementById('sub-task-template-js');
        if (!template) {
            console.error("Sub-task JS template not found!");
            return;
        }
        const clone = template.content.firstElementChild.cloneNode(true);
        clone.id = prefix;

        // Set initial static parts from data if needed, or rely on updateSubTaskUI
        clone.querySelector('[data-id="name"]').textContent = subTaskData.name;
        clone.querySelector('[data-id="name"]').title = subTaskData.name;

        // Inject progress bar HTML
        const progressBarContainer = clone.querySelector('[data-id="progress-bar-container"]');
        if (progressBarContainer) {
            progressBarContainer.innerHTML = createProgressBarHTMLForJS(prefix, subTaskData.progress, subTaskData.message, subTaskData.name);
        }
        container.appendChild(clone);
        subTaskElement = clone; // Use the newly created element
    }

    // Update all dynamic parts of the sub-task element (new or existing)
    updateTextContent(`${prefix}-name`, subTaskData.name); // Redundant if just created with name, but harmless
    updateElementClass(`${prefix}-status-badge`, subTaskData.status_class);
    updateTextContent(`${prefix}-status-badge`, subTaskData.status_text);

    // Update progress bar parts if it was created by template (or re-verify if createProgressBarHTMLForJS was called)
    // If createProgressBarHTMLForJS was used, its internal elements are already set.
    // If the progress bar container was empty and then filled by createProgressBarHTMLForJS, this call updates it.
    updateProgressUI(prefix, subTaskData.progress, subTaskData.message, subTaskData.name);

    const iconEl = subTaskElement.querySelector(`[data-id="status-icon"] i`); // Use data-id from template
    if(iconEl){
        let newIconClass = "fas text-sm ";
        if(subTaskData.status_text === 'running') newIconClass += "fa-sync fa-spin text-blue-500";
        else if(subTaskData.status_text === 'completed') newIconClass += "fa-check-circle text-green-500";
        else if(subTaskData.status_text === 'pending') newIconClass += "fa-hourglass-start text-yellow-500";
        else newIconClass += "fa-times-circle text-red-500";
        if(iconEl.className !== newIconClass) iconEl.className = newIconClass;
    }

    const subDetailsContainer = subTaskElement.querySelector(`[data-id="details-container"]`);
    const subDetailsList = subTaskElement.querySelector(`[data-id="details-list"]`);
    if(subDetailsContainer && subDetailsList) {
        if (subTaskData.details && Object.keys(subTaskData.details).length > 0) {
            subDetailsContainer.style.display = '';
            let subListContent = '';
            const sortedSubDetails = Object.entries(subTaskData.details).sort((a, b) => a[0].localeCompare(b[0]));
            for (const [key, value] of sortedSubDetails) {
                subListContent += `<li class="text-gray-500 dark:text-gray-400"><strong class="font-medium text-gray-600 dark:text-gray-300">${key}:</strong> <span class="detail-value-${subTaskData.id}-${key.replace(/[^a-zA-Z0-9-_]/g, '')}">${value}</span></li>`;
            }
            if(subDetailsList.innerHTML !== subListContent) subDetailsList.innerHTML = subListContent;
        } else {
            subDetailsContainer.style.display = 'none';
        }
    }
}


// Main function to apply updates from js_update_data
function applyJSUpdates(data) {
    if (!data) { return; }

    if (data.external_progress_bar) {
        const epb = data.external_progress_bar;
         updateProgressUI('external', epb.progress, epb.message, epb.task_name);
    }

    if (data.control_panel) {
        const cp = data.control_panel;

        if (cp.hardware_info) {
            updateHardwareInfo(cp.hardware_info);
        }

        const mainTaskData = cp.main_task;
        const mainTaskDisplayContainer = document.getElementById('main-task-name-heading'); // A parent of main task specific elements

        if (mainTaskData) {
            if(mainTaskDisplayContainer) mainTaskDisplayContainer.style.display = ''; // Ensure parent is visible
            updateTextContent('main-task-name-value', mainTaskData.name);
            updateElementClass('main-task-status-badge', mainTaskData.status_class);
            updateTextContent('main-task-status-badge', mainTaskData.status_text);
            updateProgressUI('main-task', mainTaskData.progress, mainTaskData.message, mainTaskData.name);
            updateTextContent('main-task-time-elapsed', mainTaskData.time_elapsed || 'N/A');
            updateTextContent('main-task-time-duration', mainTaskData.time_duration || 'N/A');
            updateMainTaskDetails(mainTaskData.details);
            updateMainTaskError(mainTaskData.error_message);
        } else { // No main task data, Python should have sent full HTML for "No Task"
             if(mainTaskDisplayContainer) {
                // If Python did NOT send a full HTML update and we need to hide the main task display via JS
                // This part is tricky if Python's full HTML refresh is the primary way to show "No Task"
             }
        }

        const subTasksData = cp.sub_tasks || [];
        const subTaskContainer = document.getElementById('main-task-children-container')?.querySelector('.space-y-2');

        if (subTaskContainer) {
            const existingSubTaskElements = new Map();
            subTaskContainer.querySelectorAll('.sub-task-card[id^="sub-task-"]').forEach(el => {
                existingSubTaskElements.set(el.id, el);
            });

            const dataSubTaskIds = new Set(subTasksData.map(st => `sub-task-${st.id}`));

            // Update existing or add new
            subTasksData.forEach(stData => {
                updateOrCreateSubTaskElement(stData, subTaskContainer);
                existingSubTaskElements.delete(`sub-task-${stData.id}`); // Remove from map as it's handled
            });

            // Remove old ones not in new data
            existingSubTaskElements.forEach(el => el.remove());
        }
    }
}

// Observe the hidden JSON component for changes
window.addEventListener('DOMContentLoaded', (event) => {
    const targetNode = document.getElementById('js_update_data_json');
    if (targetNode) {
        const observer = new MutationObserver(function(mutationsList, observer) {
            for(const mutation of mutationsList) {
                if (mutation.type === 'childList' || mutation.type === 'characterData' || mutation.type === 'attributes') {
                    let rawText = "";
                    const preElement = targetNode.querySelector('pre'); // Gradio JSON often has a <pre>
                    if (preElement) {
                        rawText = preElement.innerText || preElement.textContent;
                    } else {
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
                         applyJSUpdates(null);
                    }
                    break;
                }
            }
        });
        observer.observe(targetNode, { childList: true, characterData: true, subtree: true, attributes: true });
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
        const prevDisplay = contentElement.style.display;
        contentElement.style.display = 'block';
        contentElement.style.maxHeight = contentElement.scrollHeight + 'px';
        if(prevDisplay && prevDisplay !== 'block') contentElement.style.display = prevDisplay;

        contentElement.dataset.isOpen = 'true';
        chevronIcon.classList.remove('fa-chevron-down');
        chevronIcon.classList.add('fa-chevron-up');

        setTimeout(() => {
            if (contentElement.dataset.isOpen === 'true') {
                contentElement.style.maxHeight = 'none';
            }
        }, 350);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const initialTaskContent = document.getElementById('task-activity-content');
    if (initialTaskContent) {
        initialTaskContent.dataset.isOpen = 'true';
        initialTaskContent.style.opacity = '1';
        const initialChevron = document.getElementById('task-activity-chevron')?.querySelector('i');
        if (initialChevron) {
            initialChevron.classList.remove('fa-chevron-down');
            initialChevron.classList.add('fa-chevron-up');
        }
        setTimeout(() => {
            if (initialTaskContent.dataset.isOpen === 'true') {
                 if (initialTaskContent.scrollHeight > 0) {
                    initialTaskContent.style.maxHeight = initialTaskContent.scrollHeight + 'px';
                 } else {
                    initialTaskContent.style.maxHeight = '500vh';
                 }
            } else {
                initialTaskContent.style.maxHeight = '0px';
                initialTaskContent.style.opacity = '0';
            }
        }, 70);
    }
});
