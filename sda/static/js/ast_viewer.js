window.loadAndRenderAST = async function() {
    console.log('[AST Viewer] loadAndRenderAST called.');
    const container = document.getElementById('ast-visualization-container');
    if (!container) {
        console.error('[AST Viewer] Container #ast-visualization-container not found.');
        return;
    }

    const repoId = container.dataset.repoId;
    const branchName = container.dataset.branchName;
    const filePath = container.dataset.filePath;

    console.log(`[AST Viewer] Data attributes - Repo ID: ${repoId}, Branch: ${branchName}, File: ${filePath}`);

    if (!repoId || repoId === "None" || repoId === "null" || !branchName || branchName === "None" || branchName === "null" || !filePath) {
        let errorMsg = "<p>Error: Missing or invalid data attributes on container to load AST:<ul>";
        if (!repoId || repoId === "None" || repoId === "null") errorMsg += "<li>Repository ID missing</li>";
        if (!branchName || branchName === "None" || branchName === "null") errorMsg += "<li>Branch name missing</li>";
        if (!filePath) errorMsg += "<li>File path missing</li>";
        errorMsg += "</ul></p>";
        console.error("[AST Viewer] Validation of data attributes failed:", errorMsg);
        container.innerHTML = errorMsg;
        return;
    }

    // Initial loading message in container (might be set by Python already, but good to ensure)
    container.innerHTML = `Loading AST for ${filePath}...`;

    const apiUrl = `/api/repo/${repoId}/branch/${encodeURIComponent(branchName)}/file-ast?path=${encodeURIComponent(filePath)}`;
    console.log("[AST Viewer] Fetching AST data from:", apiUrl);

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText })); // Try to parse JSON, fallback to statusText
            console.error(`[AST Viewer] API Error (${response.status}):`, errorData);
            throw new Error(`API Error (${response.status}): ${errorData.detail || 'Unknown error'}`);
        }
        const astData = await response.json();

        container.innerHTML = ''; // Clear loading message
        if (astData.length === 0) {
            container.innerHTML = `<p>No AST data found for this file (it might be empty, a non-code file, or not parsed by the backend for AST).</p>`;
            console.log('[AST Viewer] No AST data returned from API.');
            return;
        }
        console.log('[AST Viewer] AST data received, rendering...');
        renderAST(astData, container);
    } catch (error) {
        console.error("[AST Viewer] Failed to load or render AST:", error);
        container.innerHTML = `<p style='color:red;'>Failed to load AST data: ${error.message}</p>`;
    }
};

function renderAST(nodes, parentElement) {
    nodes.forEach(node => {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = 'ast-node'; // For potential CSS styling from control_panel.css
        nodeDiv.style.marginLeft = (node.depth * 15) + 'px'; // Indentation based on depth
        nodeDiv.style.border = '1px dashed #ddd';
        nodeDiv.style.padding = '5px';
        nodeDiv.style.marginBottom = '5px';
        nodeDiv.style.whiteSpace = 'pre-wrap'; // Preserve whitespace and newlines in code

        // Store metadata in data attributes for tooltip
        nodeDiv.dataset.nodeType = node.type;
        nodeDiv.dataset.nodeName = node.name || 'N/A';
        nodeDiv.dataset.startLine = node.start_line;
        nodeDiv.dataset.endLine = node.end_line;

        const header = document.createElement('div');
        header.className = 'ast-node-header'; // For potential CSS styling
        header.textContent = `[${node.type}] ${node.name || ''} (L${node.start_line}-L${node.end_line})`;
        header.style.fontSize = '0.9em';
        header.style.color = '#666';
        header.style.marginBottom = '3px';
        nodeDiv.appendChild(header);

        const codeContent = document.createElement('pre'); // Use <pre> for code
        codeContent.className = 'ast-node-code'; // For potential CSS styling
        codeContent.textContent = node.code_snippet;
        codeContent.style.margin = '0'; // Reset default pre margin
        codeContent.style.padding = '0'; // Reset default pre padding
        nodeDiv.appendChild(codeContent);

        // Tooltip setup
        const tooltip = document.createElement('div');
        tooltip.className = 'ast-tooltip'; // For potential CSS styling
        tooltip.style.position = 'absolute';
        tooltip.style.visibility = 'hidden';
        tooltip.style.backgroundColor = 'black';
        tooltip.style.color = 'white';
        tooltip.style.padding = '5px';
        tooltip.style.borderRadius = '3px';
        tooltip.style.zIndex = '1000';
        tooltip.style.fontSize = '0.8em';
        tooltip.style.pointerEvents = 'none'; // So tooltip doesn't interfere
        document.body.appendChild(tooltip);

        nodeDiv.addEventListener('mouseover', (e) => {
            nodeDiv.style.backgroundColor = '#f0f0f0';
            tooltip.innerHTML = `Type: ${nodeDiv.dataset.nodeType}<br>Name: ${nodeDiv.dataset.nodeName}<br>Lines: ${nodeDiv.dataset.startLine}-${nodeDiv.dataset.endLine}`;
            tooltip.style.visibility = 'visible';
        });
        nodeDiv.addEventListener('mousemove', (e) => {
            // Position tooltip relative to the viewport
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY + 15) + 'px';
        });
        nodeDiv.addEventListener('mouseout', () => {
            nodeDiv.style.backgroundColor = '';
            tooltip.style.visibility = 'hidden';
        });

        parentElement.appendChild(nodeDiv);

        if (node.children && node.children.length > 0) {
            renderAST(node.children, nodeDiv); // Recursive call for children, append to current nodeDiv
        }
    });
}

// Optional: Add a check to see if the script is loaded
console.log('[AST Viewer] ast_viewer.js loaded.');

// It might be safer to trigger the first load attempt slightly after the main Gradio UI renders,
// or ensure the Python side calls this function explicitly after updating the HTML.
// The current plan is for Python to include a small script to call loadAndRenderAST().
// So, no auto-execution here unless specifically desired.
document.addEventListener('DOMContentLoaded', () => {
    console.log('[AST Viewer] DOMContentLoaded, attempting to set up observer.');
    // Gradio textboxes are often an input within a label, or a more complex structure.
    // The elem_id 'ast_viewer_trigger_textbox' is on a wrapper. We need to find the actual input.
    // A common structure might be a div with elem_id, containing a label, containing an input.
    // Or, if Gradio directly assigns the elem_id to the input's wrapper, querySelector might work differently.
    // Let's try to find it via the label first, as Gradio usually uses the label for querySelector targets.
    // The label for the Textbox is "ASTViewerTrigger".
    const triggerElement = document.querySelector('input[aria-label="ASTViewerTrigger"]');

    if (triggerElement) {
        console.log('[AST Viewer] AST Trigger Textbox input element found:', triggerElement);

        const observer = new MutationObserver((mutationsList, observer) => {
            // We are interested in changes to the 'value' attribute of the input
            // or if Gradio updates it by changing child nodes of a wrapper.
            // For an input, 'value' attribute change is key.
            for(const mutation of mutationsList) {
                // Check if the 'value' attribute of the input itself changed.
                if (mutation.type === 'attributes' && mutation.attributeName === 'value') {
                    console.log('[AST Viewer] AST Trigger Textbox value attribute changed. New value:', triggerElement.value);
                    if (typeof window.loadAndRenderAST === 'function') {
                        window.loadAndRenderAST();
                    } else {
                        console.error("[AST Viewer] loadAndRenderAST function not defined at time of trigger.");
                    }
                    return; // Process once per trigger
                }
            }
        });

        observer.observe(triggerElement, { attributes: true, attributeFilter: ['value'] });
        console.log('[AST Viewer] MutationObserver set up on AST Trigger Textbox input for "value" attribute.');

    } else {
        console.error('[AST Viewer] AST Trigger Textbox input element NOT found using querySelector(\'input[aria-label="ASTViewerTrigger"]\'). Visualization trigger will not work.');
        // As a fallback, try to find by elem_id if the structure is different
        const wrapperElement = document.getElementById('ast_viewer_trigger_textbox');
        if(wrapperElement) {
            console.log('[AST Viewer] Found wrapper #ast_viewer_trigger_textbox. If input not found, observer might need to target this and check childList/subtree.');
        } else {
            console.error('[AST Viewer] Wrapper #ast_viewer_trigger_textbox also not found.');
        }
    }
});
