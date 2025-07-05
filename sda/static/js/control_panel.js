// sda/static/js/control_panel.js
document.addEventListener('DOMContentLoaded', () => {
    const websocketUrl = `ws://${window.location.host}/ws/controlpanel`;
    let socket;

    // --- DOM Element Selectors (cached for performance) ---
    const modelInfo = {
        llm: document.getElementById('model-info-llm'),
        embedding: document.getElementById('model-info-embedding'),
        embeddingDevices: document.getElementById('model-info-embedding-devices')
    };
    const hardwareInfo = {
        cpus: document.getElementById('hardware-cpus'),
        cpuLoadValue: document.getElementById('cpu-load-value'),
        cpuLoadBar: document.getElementById('cpu-load-bar'),
        ramUsageValue: document.getElementById('ram-usage-value'),
        ramUsageBar: document.getElementById('ram-usage-bar'),
        gpuAvailabilityText: document.getElementById('gpu-availability-text'), // For static GPU info
        workerMaxTotal: document.getElementById('worker-max-total'),
        workerConfigList: document.getElementById('worker-config-list')
    };
    const storageInfo = {
        pgDbName: document.getElementById('pg-db-name'),
        pgSize: document.getElementById('pg-size'),
        dgraphHostPort: document.getElementById('dgraph-host-port'),
        dgraphUsage: document.getElementById('dgraph-usage')
    };
    const usageStats = {
        numRepos: document.getElementById('usage-num-repos'),
        llmCalls: document.getElementById('usage-llm-calls'),
        llmTokens: document.getElementById('usage-llm-tokens'),
        llmCost: document.getElementById('usage-llm-cost'),
        modelBreakdownContainer: document.getElementById('usage-model-breakdown')
    };

    const noActiveTaskMessageDiv = document.getElementById('no-active-task-message');
    const activeTaskWrapperDiv = document.getElementById('active-task-details-wrapper');

    // New selectors for task history and templates
    const taskHistoryList = document.getElementById('task-history-list');
    const loadMoreTasksBtn = document.getElementById('load-more-tasks-btn');
    const noHistoryMessage = document.getElementById('no-history-message');
    const taskEntryTemplate = document.getElementById('task-entry-template-js');
    const subTaskTemplate = document.getElementById('sub-task-template-js'); // Already exists, ensure it's referenced

    // State variables
    let currentRepoId = null;
    let taskHistoryOffset = 0;
    let isLoadingHistory = false;
    const TASK_HISTORY_LIMIT = 10;

    // --- START: Copied from dynamic_updates.js ---
    // (Ideally, this would be in a shared utility file if used elsewhere)
    function createProgressBarHTMLForJS(uniquePrefix, progress, message, taskName) {
        const progressText = message ? `${message} (${progress.toFixed(0)}%)` : `(${progress.toFixed(0)}%)`;
        const taskNameDisplay = taskName ? `<span id="${uniquePrefix}-task-name" class="text-xs font-semibold text-gray-700 dark:text-gray-300 truncate pr-2" title="${taskName}">${taskName}</span>` : "";
        const textContainerClass = "text-xs text-gray-600 dark:text-gray-400 mb-0.5 flex justify-between items-center";
        const progressTextSpanClass = "text-xxs whitespace-nowrap text-gray-500 dark:text-gray-300";

        return `
            <div class="progress-wrapper mb-1">
                ${taskName || message ? `
                <p class="${textContainerClass}">
                    ${taskNameDisplay}
                    <span id="${uniquePrefix}-progress-text" class="${progressTextSpanClass}">${progressText}</span>
                </p>` : `
                <p class="${textContainerClass} justify-end">
                     <span id="${uniquePrefix}-progress-text" class="${progressTextSpanClass}">${progressText}</span>
                </p>
                `}
                <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div id="${uniquePrefix}-progress-bar" class="bg-blue-500 dark:bg-blue-400 h-2 rounded-full transition-width duration-300 ease-out" style="width: ${progress}%"></div>
                </div>
            </div>
        `;
    }
    // --- END: Copied from dynamic_updates.js ---

    // const mainTaskUI object has been removed as active task rendering is now unified.
    // const subTaskTemplate = document.getElementById('sub-task-template-js'); // This template will be removed later.

    // --- Helper Functions ---
    function updateText(element, text, defaultValue = 'N/A') {
        if (element) {
            element.textContent = text || defaultValue;
        }
    }
    function updateClass(element, newClass) {
        if (element && element.className !== newClass) {
            element.className = newClass;
        }
    }
    function setVisible(element, isVisible) {
        if (element) {
            element.style.display = isVisible ? '' : 'none';
        }
    }

    function createProgressBarHTMLForJS(uniquePrefix, progress, message, taskName) {
        // Ensure text colors provide good contrast on potentially varied card backgrounds
        const progressText = message ? `${message} (${progress.toFixed(0)}%)` : `(${progress.toFixed(0)}%)`;
        const taskNameDisplay = taskName ? `<span id="${uniquePrefix}-task-name" class="text-xs font-semibold text-gray-700 dark:text-gray-300 truncate pr-2" title="${taskName}">${taskName}</span>` : "";
        // The overall <p> tag for progress text should also have good contrast.
        const textContainerClass = "text-xs text-gray-600 dark:text-gray-400 mb-0.5 flex justify-between items-center";
        const progressTextSpanClass = "text-xxs whitespace-nowrap text-gray-500 dark:text-gray-300"; // Explicit color for progress %

        return `
            <div class="progress-wrapper mb-1">
                ${taskName || message ? `
                <p class="${textContainerClass}">
                    ${taskNameDisplay}
                    <span id="${uniquePrefix}-progress-text" class="${progressTextSpanClass}">${progressText}</span>
                </p>` : `
                <p class="${textContainerClass} justify-end">
                     <span id="${uniquePrefix}-progress-text" class="${progressTextSpanClass}">${progressText}</span>
                </p>
                `}
                <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div id="${uniquePrefix}-progress-bar" class="bg-blue-500 dark:bg-blue-400 h-2 rounded-full transition-width duration-300 ease-out" style="width: ${progress}%"></div>
                </div>
            </div>
        `;
    }

    function updateProgressUI(prefix, progress, message, taskName, barElement, textElement, taskNameElement) {
        if (barElement) barElement.style.width = (progress || 0) + '%';
        if (textElement) {
            const currentProgress = progress !== null && progress !== undefined ? progress.toFixed(0) : '0';
            const currentMessage = message || "";
            const newText = `${currentMessage} (${currentProgress}%)`;
            if (textElement.innerText !== newText) textElement.innerText = newText;
        }
        if (taskNameElement) {
            const currentTaskName = taskName || "";
            if(taskNameElement.innerText !== currentTaskName) taskNameElement.innerText = currentTaskName;
            if(taskNameElement.title !== currentTaskName) taskNameElement.title = currentTaskName;
        }
    }


    // --- Data Update Functions ---
    function updateModelInfo(data) {
        if (!data) return;
        updateText(modelInfo.llm, data.active_llm_model);
        updateText(modelInfo.embedding, data.active_embedding_model);
        updateText(modelInfo.embeddingDevices, Array.isArray(data.embedding_devices) ? data.embedding_devices.join(', ') : data.embedding_devices);
    }

    function updateHardwareInfo(data) { // This data comes from control_panel_ws_data.hardware_info
        if (!data) return;
        if (hardwareInfo.cpuLoadBar) hardwareInfo.cpuLoadBar.style.width = (data.cpu_load || 0) + '%';
        updateText(hardwareInfo.cpuLoadValue, (data.cpu_load !== null ? data.cpu_load.toFixed(0) : '0') + '% Load');
        if (hardwareInfo.ramUsageBar) hardwareInfo.ramUsageBar.style.width = (data.ram_percent || 0) + '%';
        updateText(hardwareInfo.ramUsageValue, (data.ram_absolute_text || 'N/A'));

        updateText(hardwareInfo.cpus, data.num_cpus);
        updateText(hardwareInfo.workerMaxTotal, data.total_allowed_workers);

        // GPU Info
        let gpuHtml = '';
        if (data.gpu_info) {
            const gpu = data.gpu_info;
            if (gpu.torch_available && gpu.cuda_available) {
                // Ensured space after colon for GPU info
                gpuHtml = `<p class="text-xs text-gray-600 dark:text-gray-300 mb-0 leading-tight flex items-center">
                               <i class="fas fa-tv mr-1.5 text-gray-400 dark:text-gray-500"></i>
                               <strong>GPU:</strong> CUDA ${gpu.cuda_version} | ${gpu.num_gpus} Device(s)
                           </p>`;
                if (gpu.num_gpus > 0 && gpu.gpu_names && gpu.gpu_names.length > 0) {
                    gpuHtml += '<ul class="list-none pl-5 mt-0.5">';
                    gpu.gpu_names.forEach(name => {
                        gpuHtml += `<li class="text-xxs text-gray-500 dark:text-gray-400 leading-tight">${name}</li>`;
                    });
                    gpuHtml += '</ul>';
                }
            } else {
                gpuHtml = `<p class="text-xs text-gray-600 dark:text-gray-300 flex items-center"><i class="fas fa-tv-alt mr-1.5 text-gray-400 dark:text-gray-500"></i><strong>GPU:</strong> CUDA not available</p>`;
            }
        } else {
            gpuHtml = `<p class="text-xs text-gray-600 dark:text-gray-300 flex items-center"><i class="fas fa-tv-alt mr-1.5 text-gray-400 dark:text-gray-500"></i><strong>GPU:</strong> N/A</p>`;
        }
        if (hardwareInfo.gpuAvailabilityText && hardwareInfo.gpuAvailabilityText.parentElement) {
             // Assuming gpuAvailabilityText is a span inside the parent div that should be replaced
            hardwareInfo.gpuAvailabilityText.parentElement.innerHTML = gpuHtml;
        }


        // Worker Info
        let workerHtml = '';
        if (data.db_workers_per_target || data.max_embedding_workers) {
            for (const [target, num_w] of Object.entries(data.db_workers_per_target || {})) {
                workerHtml += `<li>${target.charAt(0).toUpperCase() + target.slice(1)}: ${num_w}</li>`;
            }
            if (data.max_embedding_workers !== undefined) {
                workerHtml += `<li>Embedding: ${data.max_embedding_workers}</li>`;
            }
        } else {
            workerHtml = '<li>N/A</li>';
        }
        if (hardwareInfo.workerConfigList) {
            hardwareInfo.workerConfigList.innerHTML = workerHtml;
        }
    }

    function updateStorageInfo(data) {
        if (!data) return;
        updateText(storageInfo.pgDbName, data.pg_db_name);
        updateText(storageInfo.pgSize, data.pg_size_str);
        updateText(storageInfo.dgraphHostPort, `${data.dgraph_host}:${data.dgraph_port}`);
        updateText(storageInfo.dgraphUsage, data.dgraph_usage_str);
    }

    function updateUsageStats(data) {
        if (!data) return;
        if (data.general) {
            updateText(usageStats.numRepos, data.general.num_repositories);
        }
        if (data.ai) {
            updateText(usageStats.llmCalls, data.ai.total_llm_calls);
            updateText(usageStats.llmTokens, data.ai.total_tokens_processed);
            updateText(usageStats.llmCost, data.ai.estimated_cost !== undefined ? data.ai.estimated_cost.toFixed(2) : 'N/A');
            if (data.ai.models_used && usageStats.modelBreakdownContainer) {
                let breakdownHtml = '<ul class="list-disc list-inside pl-4 mt-1 text-xxs">';
                for (const [model, stats] of Object.entries(data.ai.models_used)) {
                    breakdownHtml += `<li><strong>${model}:</strong> ${stats.calls} calls, ${stats.tokens} tokens, $${stats.cost.toFixed(2)}</li>`;
                }
                breakdownHtml += '</ul>';
                usageStats.modelBreakdownContainer.innerHTML = breakdownHtml;
            }
        }
    }

    // updateMainTask function removed as its functionality is merged into the WebSocket onmessage handler
    // and uses renderTaskEntry for the active task.

    // updateOrCreateSubTaskElement function removed as sub-tasks will be rendered by renderTaskEntry.
    // updateSubTasks function removed for the same reason.

    // --- WebSocket Connection ---
    function connect() {
        socket = new WebSocket(websocketUrl);

        socket.onopen = () => {
            console.log('Control Panel WebSocket connected.');
            // You could send an initial message if needed, e.g., client identity
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // console.log('Control Panel WebSocket data received:', data);

                // Try to get repo_id from the payload. This needs to be sent by the backend.
                // For now, we assume 'current_repo_id' might be a key in 'data' or 'data.system_info'
                let newRepoId = data.current_repo_id || (data.system_info ? data.system_info.current_repo_id : null);

                if (newRepoId && newRepoId !== currentRepoId) {
                    currentRepoId = newRepoId;
                    fetchAndRenderHistoricalTasks(true); // Fetch history for the new repo
                } else if (!newRepoId && currentRepoId) {
                    // Repo became null (e.g. no repo selected), clear history
                    currentRepoId = null;
                    if(taskHistoryList) taskHistoryList.innerHTML = '';
                    if(noHistoryMessage) {
                        noHistoryMessage.textContent = 'No repository selected to show task history.';
                        noHistoryMessage.style.display = 'block';
                    }
                    if(loadMoreTasksBtn) loadMoreTasksBtn.disabled = true;
                }


                if (data.main_task !== undefined && data.main_task !== null) {
                    setVisible(noActiveTaskMessageDiv, false);
                    setVisible(activeTaskWrapperDiv, true);
                    activeTaskWrapperDiv.innerHTML = ''; // Clear previous active task content

                    // Render the main task using the unified renderTaskEntry
                    const activeTaskElement = renderTaskEntry(data.main_task, true); // true for isExpanded
                    if (activeTaskElement) {
                        activeTaskWrapperDiv.appendChild(activeTaskElement);
                    } else {
                        // Fallback if rendering fails, though renderTaskEntry should handle null gracefully.
                        setVisible(noActiveTaskMessageDiv, true);
                        setVisible(activeTaskWrapperDiv, false);
                        if(noActiveTaskMessageDiv) noActiveTaskMessageDiv.textContent = "Error rendering active task.";
                    }
                } else {
                    // No main_task in the message, or it's null
                    setVisible(noActiveTaskMessageDiv, true);
                    setVisible(activeTaskWrapperDiv, false);
                    activeTaskWrapperDiv.innerHTML = ''; // Clear if no active task
                    if(noActiveTaskMessageDiv) { // Ensure the message is appropriate
                         noActiveTaskMessageDiv.querySelector('p.font-medium').textContent = "No Active Task";
                         noActiveTaskMessageDiv.querySelector('p.text-sm').textContent = "The system is currently idle or no repository is selected.";
                    }
                }

                // Sub-tasks are now rendered recursively by renderTaskEntry if they exist in data.main_task.children.
                // The old updateSubTasks function and its direct calls are removed.

                if (data.hardware_info) {
                    updateHardwareInfo(data.hardware_info);
                }
                if (data.system_info) {
                    if(data.system_info.model_info) updateModelInfo(data.system_info.model_info);
                    if(data.system_info.storage_info) updateStorageInfo(data.system_info.storage_info);
                    if(data.system_info.usage_stats) updateUsageStats(data.system_info.usage_stats);
                    // If repo_id is part of system_info and wasn't caught above
                    if (data.system_info.current_repo_id && data.system_info.current_repo_id !== currentRepoId) {
                        currentRepoId = data.system_info.current_repo_id;
                        fetchAndRenderHistoricalTasks(true);
                    }
                }
            } catch (e) {
                console.error('Error processing message from Control Panel WebSocket:', e);
            }
        };

        socket.onclose = (event) => {
            console.log('Control Panel WebSocket disconnected. Attempting to reconnect...', event.reason);
            setTimeout(connect, 3000); // Reconnect after 3 seconds
        };

        socket.onerror = (error) => {
            console.error('Control Panel WebSocket error:', error);
            // The onclose event will usually fire after an error, triggering reconnection.
        };
    }

    // Initial connection
    connect();

    // --- Task History Fetching and Rendering ---
    async function fetchAndRenderHistoricalTasks(clearExisting = false) {
        if (isLoadingHistory || !currentRepoId) {
            if (!currentRepoId) console.log("Cannot fetch history, currentRepoId is not set.");
            return;
        }
        isLoadingHistory = true;
        if(loadMoreTasksBtn) loadMoreTasksBtn.disabled = true;

        if (clearExisting) {
            taskHistoryOffset = 0;
            if(taskHistoryList) taskHistoryList.innerHTML = '';
            if(noHistoryMessage) noHistoryMessage.style.display = 'block'; // Show initially
        }

        try {
            const response = await fetch(`/api/repositories/${currentRepoId}/tasks_history?offset=${taskHistoryOffset}&limit=${TASK_HISTORY_LIMIT}`);
            if (!response.ok) {
                console.error(`Error fetching task history: ${response.status} ${response.statusText}`);
                if(noHistoryMessage && taskHistoryList.children.length === 0) noHistoryMessage.textContent = 'Failed to load task history.';
                return;
            }
            const tasks = await response.json();

            if (tasks && tasks.length > 0) {
                if(noHistoryMessage) noHistoryMessage.style.display = 'none';
                tasks.forEach(task => {
                    const taskElement = renderTaskEntry(task, false); // Historical tasks collapsed by default
                    if (taskElement && taskHistoryList) taskHistoryList.appendChild(taskElement);
                });
                taskHistoryOffset += tasks.length;
                if(loadMoreTasksBtn) loadMoreTasksBtn.disabled = tasks.length < TASK_HISTORY_LIMIT;
            } else {
                if(loadMoreTasksBtn) loadMoreTasksBtn.disabled = true; // No more tasks or no tasks at all
                if(taskHistoryList.children.length === 0 && noHistoryMessage) {
                    noHistoryMessage.textContent = 'No historical tasks found.';
                    noHistoryMessage.style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Exception fetching task history:', error);
            if(noHistoryMessage && taskHistoryList.children.length === 0) noHistoryMessage.textContent = 'Error loading task history.';
        } finally {
            isLoadingHistory = false;
            // Re-enable button only if there might be more tasks (not strictly necessary here as it's set based on fetch results)
            if (loadMoreTasksBtn && !loadMoreTasksBtn.disabled) {
                 // Check if it was disabled due to fetch results, not just isLoadingHistory
            }
        }
    }

    if (loadMoreTasksBtn) {
        loadMoreTasksBtn.addEventListener('click', () => fetchAndRenderHistoricalTasks(false));
    }


    // --- Rendering Functions ---
    // renderSubTaskEntry function removed as sub-tasks are now rendered by renderTaskEntry using the main task template.

    function renderTaskEntry(taskData, isExpanded = false) {
        if (!taskData || !taskData.id) { // Added a check for taskData and taskData.id
             console.warn("renderTaskEntry called with invalid taskData:", taskData);
             return null; // Return null if taskData is not valid
        }
        if (!taskEntryTemplate) {
            console.error("Task entry template not found!");
            return null;
        }
        const clone = taskEntryTemplate.content.firstElementChild.cloneNode(true);
        const prefix = `task-entry-${taskData.id}`;
        clone.id = prefix;

        // Task Name and Title
        const taskNameHeading = clone.querySelector('.task-name-heading');
        if (taskNameHeading) {
            const taskNamePlaceholder = taskNameHeading.querySelector('.task-name-placeholder');
            if (taskNamePlaceholder) taskNamePlaceholder.textContent = taskData.name || 'Unnamed Task';
            taskNameHeading.title = taskData.name || 'Unnamed Task';
        }


        // Status Badge
        const statusBadge = clone.querySelector('.task-status-badge');
        if (statusBadge) {
            statusBadge.className = taskData.status_class || 'px-3 py-1 text-xs font-semibold rounded-full bg-gray-200 text-gray-800 dark:bg-gray-600 dark:text-gray-200';
            statusBadge.textContent = taskData.status_text || 'Unknown';
        }

        // Task Icon
        const taskIcon = clone.querySelector('.task-icon');
        if (taskIcon) {
            if (taskData.status_text === 'running') taskIcon.className = 'task-icon fas fa-sync fa-spin mr-1.5 text-blue-500';
            else if (taskData.status_text === 'completed') taskIcon.className = 'task-icon fas fa-check-circle mr-1.5 text-green-500';
            else if (taskData.status_text === 'failed') taskIcon.className = 'task-icon fas fa-times-circle mr-1.5 text-red-500';
            else if (taskData.status_text === 'pending') taskIcon.className = 'task-icon fas fa-hourglass-half mr-1.5 text-yellow-500';
            else taskIcon.className = 'task-icon fas fa-flag-checkered mr-1.5 text-indigo-500'; // Default/fallback
        }

        // Progress Bar
        const progressBarContainer = clone.querySelector('.task-progress-bar-container');
        if (progressBarContainer) {
            if (taskData.status_text === 'running' || taskData.status_text === 'pending') {
                progressBarContainer.innerHTML = createProgressBarHTMLForJS(
                    prefix, taskData.progress, taskData.message, taskData.name
                );
                progressBarContainer.style.display = 'block';
            } else {
                progressBarContainer.style.display = 'none';
            }
        }

        // Timing Information
        const timingPlaceholder = clone.querySelector('.task-timing-placeholder');
        if (timingPlaceholder) {
            let timingHtml = '';
            if (taskData.time_elapsed) timingHtml += `<span class="mr-2"><i class="far fa-clock mr-1"></i>Elapsed: ${taskData.time_elapsed}</span>`;
            if (taskData.time_duration) timingHtml += `<span><i class="fas fa-stopwatch mr-1"></i>Duration: ${taskData.time_duration}</span>`;
            timingPlaceholder.innerHTML = timingHtml || 'Timing N/A';
        }

        // Details Section
        const detailsCard = clone.querySelector('.task-details-card');
        const detailsList = clone.querySelector('.task-details-list');
        if (detailsCard && detailsList) {
            if (taskData.details && Object.keys(taskData.details).length > 0) {
                let listContent = '';
                // Sort details by key for consistent order
                const sortedDetails = Object.entries(taskData.details).sort((a,b) => a[0].localeCompare(b[0]));
                for (const [key, value] of sortedDetails) {
                    listContent += `<li><strong>${key}:</strong> ${value}</li>`;
                }
                detailsList.innerHTML = listContent;
                detailsCard.style.display = 'block';
            } else {
                detailsCard.style.display = 'none';
            }
        }

        // Error Section
        const errorCard = clone.querySelector('.task-error-card');
        const errorMessageContent = clone.querySelector('.task-error-message-content');
        if (errorCard && errorMessageContent) {
            if (taskData.error_message) {
                errorMessageContent.textContent = taskData.error_message; // Using textContent for potentially multi-line errors
                errorCard.style.display = 'block';
            } else {
                errorCard.style.display = 'none';
            }
        }

        // Sub-Tasks (Children) Section
        const subTasksListContainer = clone.querySelector('.sub-tasks-list');
        const childrenContainer = clone.querySelector('.task-children-container');
        if (subTasksListContainer && childrenContainer) {
            subTasksListContainer.innerHTML = ''; // Clear any existing sub-tasks from template
            if (taskData.children && taskData.children.length > 0) {
                taskData.children.forEach(childTaskData => {
                    // Recursively call renderTaskEntry for each child.
                    // Child tasks are typically not expanded by default.
                    const childTaskElement = renderTaskEntry(childTaskData, false);
                    if (childTaskElement) {
                        // Add a class to identify sub-task entries for styling if needed
                        childTaskElement.classList.add('sub-task-display');
                        subTasksListContainer.appendChild(childTaskElement);
                    }
                });
                childrenContainer.style.display = 'block';
            } else {
                childrenContainer.style.display = 'none';
            }
        }

        // Collapsible Content
        const summaryDiv = clone.querySelector('.task-summary');
        const collapsibleContent = clone.querySelector('.task-content-collapsible');

        if (summaryDiv && collapsibleContent) { // Ensure elements exist
            if (isExpanded) {
                collapsibleContent.style.display = 'block';
            } else {
                collapsibleContent.style.display = 'none'; // Ensure it's hidden if not expanded
            }

            summaryDiv.addEventListener('click', () => {
                const currentlyExpanded = collapsibleContent.style.display === 'block';
                collapsibleContent.style.display = currentlyExpanded ? 'none' : 'block';
                // TODO: Add chevron icon to summaryDiv and rotate it (existing TODO)
            });
        } else {
            console.warn("Task summary or collapsible content not found in template for task:", taskData.id);
        }

        return clone;
    }


    // Collapsible section logic (from dynamic_updates.js, adapted slightly)
    const taskActivityToggle = document.getElementById('task-activity-toggle');
    const taskActivityContent = document.getElementById('task-activity-content');
    const taskActivityChevron = document.getElementById('task-activity-chevron')?.querySelector('i');

    if (taskActivityToggle && taskActivityContent && taskActivityChevron) {
        // Initialize open
        taskActivityContent.style.maxHeight = taskActivityContent.scrollHeight + 'px';
        taskActivityContent.style.opacity = '1';
        taskActivityContent.dataset.isOpen = 'true';
        taskActivityChevron.classList.remove('fa-chevron-down'); // Corrected variable name
        taskActivityChevron.classList.add('fa-chevron-up');     // Corrected variable name
         setTimeout(() => { // Ensure scrollHeight is calculated after full render
            if (taskActivityContent.dataset.isOpen === 'true') {
                 taskActivityContent.style.maxHeight = 'none'; // Allow dynamic content
            }
        }, 350);


        taskActivityToggle.addEventListener('click', () => {
            const isOpen = taskActivityContent.dataset.isOpen === 'true';
            if (isOpen) {
                taskActivityContent.style.maxHeight = '0px';
                taskActivityContent.style.opacity = '0';
                taskActivityContent.dataset.isOpen = 'false';
                taskActivityChevron.classList.remove('fa-chevron-up');
                taskActivityChevron.classList.add('fa-chevron-down');
            } else {
                taskActivityContent.style.display = 'block'; // Ensure it's block for scrollHeight calc
                taskActivityContent.style.maxHeight = taskActivityContent.scrollHeight + 'px';
                taskActivityContent.style.opacity = '1';
                taskActivityContent.dataset.isOpen = 'true';
                taskActivityChevron.classList.remove('fa-chevron-down');
                taskActivityChevron.classList.add('fa-chevron-up');
                setTimeout(() => {
                    if (taskActivityContent.dataset.isOpen === 'true') {
                        taskActivityContent.style.maxHeight = 'none'; // Allow dynamic content to expand
                    }
                }, 350); // Match transition duration
            }
        });
    }
});
