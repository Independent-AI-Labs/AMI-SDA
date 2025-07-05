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
    const activeTaskWrapperDiv = document.getElementById('active-task-details-wrapper'); // This might be repurposed or removed if active task is part of the new list

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


    const mainTaskUI = { // This structure might be less used if active task is rendered via template
        name: document.getElementById('main-task-name'),
        status: document.getElementById('main-task-status'),
        progressText: document.getElementById('main-task-progress-text'),
        progressBar: document.getElementById('main-task-progress-bar'),
        taskNameDisplay: document.getElementById('main-task-task-name'), // For progress bar's task name
        timeElapsed: document.getElementById('main-task-time-elapsed'),
        timeDuration: document.getElementById('main-task-time-duration'),
        detailsCard: document.getElementById('main-task-details-card'),
        detailsList: document.getElementById('main-task-details-list'),
        errorCard: document.getElementById('main-task-error-card'),
        errorMessageContent: document.getElementById('main-task-error-message-content'),
        childrenContainer: document.getElementById('main-task-children-container'),
        subTasksListJS: document.getElementById('sub-tasks-list-js') // Container for JS-generated sub-tasks
    };
    const subTaskTemplate = document.getElementById('sub-task-template-js');

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

    function updateMainTask(taskData) {
        if (!taskData) {
            setVisible(noActiveTaskMessageDiv, true);
            setVisible(activeTaskWrapperDiv, false);
            return;
        }
        setVisible(noActiveTaskMessageDiv, false);
        setVisible(activeTaskWrapperDiv, true);

        updateText(mainTaskUI.name, taskData.name);
        updateClass(mainTaskUI.status, taskData.status_class);
        updateText(mainTaskUI.status, taskData.status_text);

        updateProgressUI('main-task', taskData.progress, taskData.message, taskData.name,
            mainTaskUI.progressBar, mainTaskUI.progressText, mainTaskUI.taskNameDisplay);

        updateText(mainTaskUI.timeElapsed, taskData.time_elapsed);
        updateText(mainTaskUI.timeDuration, taskData.time_duration);

        // Details
        if (taskData.details && Object.keys(taskData.details).length > 0) {
            let listContent = '';
            const sortedDetails = Object.entries(taskData.details).sort((a, b) => a[0].localeCompare(b[0]));
            for (const [key, value] of sortedDetails) {
                listContent += `<li class="text-xs text-gray-600 dark:text-gray-400"><strong class="font-medium text-gray-700 dark:text-gray-300">${key}:</strong> <span class="detail-value">${value}</span></li>`;
            }
            mainTaskUI.detailsList.innerHTML = listContent;
            setVisible(mainTaskUI.detailsCard, true);
        } else {
            setVisible(mainTaskUI.detailsCard, false);
        }

        // Error
        if (taskData.error_message) {
            updateText(mainTaskUI.errorMessageContent, taskData.error_message);
            setVisible(mainTaskUI.errorCard, true);
        } else {
            setVisible(mainTaskUI.errorCard, false);
        }
    }

    function updateOrCreateSubTaskElement(subTaskData, container) {
        const prefix = `sub-task-${subTaskData.id}`;
        let subTaskElement = document.getElementById(prefix);

        if (!subTaskElement) {
            if (!subTaskTemplate) { console.error("Sub-task JS template not found!"); return; }
            const clone = subTaskTemplate.content.firstElementChild.cloneNode(true);
            clone.id = prefix;

            clone.querySelector('[data-id="name"]').textContent = subTaskData.name;
            clone.querySelector('[data-id="name"]').title = subTaskData.name;

            const progressBarContainer = clone.querySelector('[data-id="progress-bar-container"]');
            if (progressBarContainer) {
                progressBarContainer.innerHTML = createProgressBarHTMLForJS(prefix, subTaskData.progress, subTaskData.message, subTaskData.name);
            }
            container.appendChild(clone);
            subTaskElement = clone;
        }

        // Update dynamic parts
        updateText(subTaskElement.querySelector(`[data-id="name"]`), subTaskData.name);
        subTaskElement.querySelector(`[data-id="name"]`).title = subTaskData.name;
        updateClass(subTaskElement.querySelector(`[data-id="status-badge"]`), subTaskData.status_class);
        updateText(subTaskElement.querySelector(`[data-id="status-badge"]`), subTaskData.status_text);

        updateProgressUI(prefix, subTaskData.progress, subTaskData.message, subTaskData.name,
            subTaskElement.querySelector(`#${prefix}-progress-bar`),
            subTaskElement.querySelector(`#${prefix}-progress-text`),
            subTaskElement.querySelector(`#${prefix}-task-name`)
        );

        const iconEl = subTaskElement.querySelector(`[data-id="status-icon"] i`);
        if(iconEl){
            let newIconClass = "fas text-sm ";
            if(subTaskData.status_text === 'running') newIconClass += "fa-sync fa-spin text-blue-500";
            else if(subTaskData.status_text === 'completed') newIconClass += "fa-check-circle text-green-500";
            else if(subTaskData.status_text === 'pending') newIconClass += "fa-hourglass-start text-yellow-500";
            else newIconClass += "fa-times-circle text-red-500"; // failed or other
            if(iconEl.className !== newIconClass) iconEl.className = newIconClass;
        }

        const subDetailsContainer = subTaskElement.querySelector(`[data-id="details-container"]`);
        const subDetailsList = subTaskElement.querySelector(`[data-id="details-list"]`);
        if(subDetailsContainer && subDetailsList) {
            if (subTaskData.details && Object.keys(subTaskData.details).length > 0) {
                let subListContent = '';
                const sortedSubDetails = Object.entries(subTaskData.details).sort((a,b) => a[0].localeCompare(b[0]));
                for (const [key, value] of sortedSubDetails) {
                     subListContent += `<li class="text-gray-500 dark:text-gray-400"><strong class="font-medium text-gray-600 dark:text-gray-300">${key}:</strong> ${value}</li>`;
                }
                subDetailsList.innerHTML = subListContent;
                setVisible(subDetailsContainer, true);
            } else {
                setVisible(subDetailsContainer, false);
            }
        }
    }

    function updateSubTasks(subTasksData = []) {
        if (!mainTaskUI.subTasksListJS) return;
        setVisible(mainTaskUI.childrenContainer, subTasksData.length > 0);

        const existingSubTaskElements = new Map();
        mainTaskUI.subTasksListJS.querySelectorAll('.sub-task-card[id^="sub-task-"]').forEach(el => {
            existingSubTaskElements.set(el.id, el);
        });

        subTasksData.forEach(stData => {
            updateOrCreateSubTaskElement(stData, mainTaskUI.subTasksListJS);
            existingSubTaskElements.delete(`sub-task-${stData.id}`);
        });

        existingSubTaskElements.forEach(el => el.remove());
    }

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


                if (data.main_task !== undefined) {
                    setVisible(noActiveTaskMessageDiv, false);
                    setVisible(activeTaskWrapperDiv, true);
                    activeTaskWrapperDiv.innerHTML = ''; // Clear previous active task
                    const activeTaskElement = renderTaskEntry(data.main_task, true); // Active task is expanded
                    if (activeTaskElement) {
                        activeTaskWrapperDiv.appendChild(activeTaskElement);
                    }
                } else {
                     setVisible(noActiveTaskMessageDiv, true);
                     setVisible(activeTaskWrapperDiv, false);
                     activeTaskWrapperDiv.innerHTML = ''; // Clear if no active task
                }

                // The old updateSubTasks was tied to the old mainTaskUI structure.
                // Sub-tasks are now rendered as part of renderTaskEntry.

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
    function renderSubTaskEntry(subTaskData) {
        if (!subTaskTemplate) {
            console.error("Sub-task template not found!");
            return null;
        }
        const clone = subTaskTemplate.content.firstElementChild.cloneNode(true);
        const prefix = `sub-task-entry-${subTaskData.id}`; // Unique prefix for elements within this sub-task
        clone.id = prefix;

        clone.querySelector('.subtask-name').textContent = subTaskData.name;
        clone.querySelector('.subtask-name').title = subTaskData.name;

        const statusBadge = clone.querySelector('.subtask-status-badge');
        statusBadge.className = subTaskData.status_class || 'text-xxs px-1.5 py-0.5 rounded-full whitespace-nowrap flex-shrink-0 bg-gray-200 text-gray-800';
        statusBadge.textContent = subTaskData.status_text || 'Unknown';

        const iconEl = clone.querySelector(`.subtask-status-icon i`);
        if(iconEl){
            let newIconClass = "fas text-xs ";
            if(subTaskData.status_text === 'running') newIconClass += "fa-sync fa-spin text-blue-500";
            else if(subTaskData.status_text === 'completed') newIconClass += "fa-check-circle text-green-500";
            else if(subTaskData.status_text === 'pending') newIconClass += "fa-hourglass-start text-yellow-500";
            else newIconClass += "fa-times-circle text-red-500"; // failed or other
            iconEl.className = newIconClass;
        }

        const progressBarContainer = clone.querySelector('.subtask-progress-bar-container');
        if (subTaskData.status_text === 'running' || subTaskData.status_text === 'pending') {
            progressBarContainer.innerHTML = createProgressBarHTMLForJS(prefix, subTaskData.progress, subTaskData.message, ""); // No task name for sub-task progress bar
            progressBarContainer.style.display = '';
        } else {
            progressBarContainer.style.display = 'none';
        }

        const detailsContainer = clone.querySelector('.subtask-details-container');
        const detailsList = clone.querySelector('.subtask-details-list');
        if (subTaskData.details && Object.keys(subTaskData.details).length > 0) {
            let listContent = '';
            const sortedSubDetails = Object.entries(subTaskData.details).sort((a,b) => a[0].localeCompare(b[0]));
            for (const [key, value] of sortedSubDetails) {
                 listContent += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            detailsList.innerHTML = listContent;
            detailsContainer.style.display = '';
        } else {
            detailsContainer.style.display = 'none';
        }

        const summaryDiv = clone.querySelector('.sub-task-summary');
        const collapsibleContent = clone.querySelector('.sub-task-content-collapsible');
        summaryDiv.addEventListener('click', () => {
            const isExpanded = collapsibleContent.style.display === 'block';
            collapsibleContent.style.display = isExpanded ? 'none' : 'block';
            // Optionally, add a chevron icon and rotate it
        });
        return clone;
    }

    function renderTaskEntry(taskData, isExpanded = false) {
        if (!taskEntryTemplate) {
            console.error("Task entry template not found!");
            return null;
        }
        const clone = taskEntryTemplate.content.firstElementChild.cloneNode(true);
        const prefix = `task-entry-${taskData.id}`;
        clone.id = prefix;

        clone.querySelector('.task-name-placeholder').textContent = taskData.name;
        clone.querySelector('.task-name-heading').title = taskData.name; // For long names

        const statusBadge = clone.querySelector('.task-status-badge');
        statusBadge.className = taskData.status_class || 'px-3 py-1 text-xs font-semibold rounded-full bg-gray-200 text-gray-800';
        statusBadge.textContent = taskData.status_text || 'Unknown';

        const taskIcon = clone.querySelector('.task-icon');
        if (taskData.status_text === 'running') taskIcon.className = 'task-icon fas fa-sync fa-spin mr-2 text-blue-500';
        else if (taskData.status_text === 'completed') taskIcon.className = 'task-icon fas fa-check-circle mr-2 text-green-500';
        else if (taskData.status_text === 'failed') taskIcon.className = 'task-icon fas fa-times-circle mr-2 text-red-500';
        else if (taskData.status_text === 'pending') taskIcon.className = 'task-icon fas fa-hourglass-half mr-2 text-yellow-500';
        else taskIcon.className = 'task-icon fas fa-flag-checkered mr-2 text-indigo-500';


        const progressBarContainer = clone.querySelector('.task-progress-bar-container');
        if (taskData.status_text === 'running' || taskData.status_text === 'pending') {
            progressBarContainer.innerHTML = createProgressBarHTMLForJS(
                prefix, taskData.progress, taskData.message, taskData.name
            );
            progressBarContainer.style.display = 'block';
        } else {
            progressBarContainer.style.display = 'none';
        }

        const timingPlaceholder = clone.querySelector('.task-timing-placeholder');
        let timingHtml = '';
        if (taskData.time_elapsed) timingHtml += `<span class="mr-2"><i class="far fa-clock mr-1"></i>Elapsed: ${taskData.time_elapsed}</span>`;
        if (taskData.time_duration) timingHtml += `<span><i class="fas fa-stopwatch mr-1"></i>Duration: ${taskData.time_duration}</span>`;
        timingPlaceholder.innerHTML = timingHtml || 'Timing N/A';


        const detailsCard = clone.querySelector('.task-details-card');
        const detailsList = clone.querySelector('.task-details-list');
        if (taskData.details && Object.keys(taskData.details).length > 0) {
            let listContent = '';
            const sortedDetails = Object.entries(taskData.details).sort((a,b) => a[0].localeCompare(b[0]));
            for (const [key, value] of sortedDetails) {
                listContent += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            detailsList.innerHTML = listContent;
            detailsCard.style.display = 'block';
        } else {
            detailsCard.style.display = 'none';
        }

        const errorCard = clone.querySelector('.task-error-card');
        const errorMessageContent = clone.querySelector('.task-error-message-content');
        if (taskData.error_message) {
            errorMessageContent.textContent = taskData.error_message;
            errorCard.style.display = 'block';
        } else {
            errorCard.style.display = 'none';
        }

        const subTasksListContainer = clone.querySelector('.sub-tasks-list');
        const childrenContainer = clone.querySelector('.task-children-container');
        if (taskData.children && taskData.children.length > 0) {
            taskData.children.forEach(subTask => {
                const subTaskElement = renderSubTaskEntry(subTask);
                if (subTaskElement) subTasksListContainer.appendChild(subTaskElement);
            });
            childrenContainer.style.display = 'block';
        } else {
            childrenContainer.style.display = 'none';
        }

        const summaryDiv = clone.querySelector('.task-summary');
        const collapsibleContent = clone.querySelector('.task-content-collapsible');

        if (isExpanded) {
            collapsibleContent.style.display = 'block';
        }

        summaryDiv.addEventListener('click', () => {
            const currentlyExpanded = collapsibleContent.style.display === 'block';
            collapsibleContent.style.display = currentlyExpanded ? 'none' : 'block';
            // TODO: Add chevron icon to summaryDiv and rotate it
        });

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
