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

    const mainTaskUI = {
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
                console.log('Control Panel WebSocket data received:', data); // ADDED FOR DEBUGGING

                if (data.main_task !== undefined) { // Check for presence of main_task key
                    updateMainTask(data.main_task);
                }
                if (data.sub_tasks) {
                    updateSubTasks(data.sub_tasks);
                }
                if (data.hardware_info) { // This comes from control_panel_ws_data.hardware_info
                    updateHardwareInfo(data.hardware_info);
                }
                // Assuming system_info (model, storage, usage) comes in a different structure or top-level keys
                // For now, let's assume they are part of a 'system_info' key in the WS payload for simplicity.
                if (data.system_info) {
                    if(data.system_info.model_info) updateModelInfo(data.system_info.model_info);
                    if(data.system_info.storage_info) updateStorageInfo(data.system_info.storage_info);
                    if(data.system_info.usage_stats) updateUsageStats(data.system_info.usage_stats);
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
