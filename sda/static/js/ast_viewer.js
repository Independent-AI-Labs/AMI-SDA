window.initializeASTViewer = async function(params) {
    console.log('[AST Viewer] initializeASTViewer called with params:', params);
    const displayArea = document.getElementById('ast-display-area'); // Target div in ast_visualization.html

    if (!displayArea) {
        console.error('[AST Viewer] Container #ast-display-area not found.');
        return;
    }

    if (!params || !params.repoId || !params.branchName || !params.filePath) {
        let errorMsg = "<p style='color:red;'>Error: Missing necessary parameters to load AST.<ul>";
        if (!params.repoId) errorMsg += "<li>Repository ID missing</li>";
        if (!params.branchName) errorMsg += "<li>Branch name missing</li>";
        if (!params.filePath) errorMsg += "<li>File path missing</li>";
        errorMsg += "</ul></p>";
        console.error("[AST Viewer] Parameter validation failed:", errorMsg);
        displayArea.innerHTML = errorMsg;
        return;
    }

    const { repoId, branchName, filePath } = params;

    // Set initial loading message in the display area
    displayArea.innerHTML = `Loading AST for ${filePath}...`;
    console.log(`[AST Viewer] Params - Repo ID: ${repoId}, Branch: ${branchName}, File: ${filePath}`);

    // Clear previous listeners if any, and reset state
    if (window.astViewerGlobalState && window.astViewerGlobalState.displayArea) {
        const oldArea = window.astViewerGlobalState.displayArea;
        oldArea.removeEventListener('mouseover', window.astViewerGlobalState.mouseoverHandler);
        oldArea.removeEventListener('mouseout', window.astViewerGlobalState.mouseoutHandler);
        oldArea.removeEventListener('mousemove', window.astViewerGlobalState.mousemoveHandler);
    }
    window.astViewerGlobalState = {
        currentlyHighlightedNode: null,
        tooltipElement: null,
        displayArea: displayArea // Store reference for cleanup
    };

    // Create or get tooltip
    let tooltip = document.querySelector('.ast-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'ast-tooltip';
        // Styles are in ast_visualization.html or a shared CSS, keep JS for dynamic properties
        tooltip.style.position = 'absolute'; // JS needs to control this for positioning
        tooltip.style.visibility = 'hidden';
        tooltip.style.zIndex = '1000'; // Ensure it's on top
        tooltip.style.pointerEvents = 'none';
        document.body.appendChild(tooltip);
    }
    window.astViewerGlobalState.tooltipElement = tooltip;


    // Event Handlers for delegation
    window.astViewerGlobalState.mouseoverHandler = function(event) {
        const targetNode = event.target.closest('.ast-node');
        let state = window.astViewerGlobalState;

        if (state.currentlyHighlightedNode && state.currentlyHighlightedNode !== targetNode) {
            state.currentlyHighlightedNode.classList.remove('ast-node-highlighted');
            // state.currentlyHighlightedNode.style.backgroundColor = '#ffffff'; // Revert to default if not using class for default
        }

        if (targetNode) {
            targetNode.classList.add('ast-node-highlighted');
            // targetNode.style.backgroundColor = '#f0f0f0'; // If not using class for highlight
            state.currentlyHighlightedNode = targetNode;

            // Update and show tooltip
            let tooltipContent = `Type: ${targetNode.dataset.nodeType}<br>Name: ${targetNode.dataset.nodeName}<br>Lines: L${targetNode.dataset.startLine}${targetNode.dataset.startColumn !== 'N/A' ? ':' + targetNode.dataset.startColumn : ''} - L${targetNode.dataset.endLine}${targetNode.dataset.endColumn !== 'N/A' ? ':' + targetNode.dataset.endColumn : ''}<br>Tokens: ${targetNode.dataset.tokenCount}<br>Children: ${targetNode.dataset.childrenCount}<br>Degree: ${targetNode.dataset.dgraphDegree}`;
            state.tooltipElement.innerHTML = tooltipContent;
            state.tooltipElement.style.visibility = 'visible';
        }
    };

    window.astViewerGlobalState.mouseoutHandler = function(event) {
        const targetNode = event.target.closest('.ast-node');
        let state = window.astViewerGlobalState;

        // If the mouse is leaving a node AND not entering another .ast-node that is a child of it
        if (targetNode && (!event.relatedTarget || !targetNode.contains(event.relatedTarget.closest && event.relatedTarget.closest('.ast-node')))) {
             if (state.currentlyHighlightedNode === targetNode) { // Ensure we are de-highlighting the correct one
                targetNode.classList.remove('ast-node-highlighted');
                // targetNode.style.backgroundColor = '#ffffff'; // Revert to default if not using class for default
                state.currentlyHighlightedNode = null;
                state.tooltipElement.style.visibility = 'hidden';
            }
        } else if (!targetNode && state.currentlyHighlightedNode) {
            // This case handles moving out of the displayArea entirely or to a non-node element within it
             state.currentlyHighlightedNode.classList.remove('ast-node-highlighted');
             state.currentlyHighlightedNode = null;
             state.tooltipElement.style.visibility = 'hidden';
        }
    };

    window.astViewerGlobalState.mousemoveHandler = function(event) {
        let state = window.astViewerGlobalState;
        if (state.tooltipElement && state.tooltipElement.style.visibility === 'visible') {
            state.tooltipElement.style.left = (event.clientX + 15) + 'px';
            state.tooltipElement.style.top = (event.clientY + 15) + 'px';
        }
    };

    displayArea.addEventListener('mouseover', window.astViewerGlobalState.mouseoverHandler);
    displayArea.addEventListener('mouseout', window.astViewerGlobalState.mouseoutHandler);
    document.body.addEventListener('mousemove', window.astViewerGlobalState.mousemoveHandler); // Tooltip position relative to body/viewport


    const apiUrl = `/api/repo/${repoId}/branch/${encodeURIComponent(branchName)}/file-ast?path=${encodeURIComponent(filePath)}`;
    console.log("[AST Viewer] Fetching AST data from:", apiUrl);

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            console.error(`[AST Viewer] API Error (${response.status}):`, errorData);
            throw new Error(`API Error (${response.status}): ${errorData.detail || 'Unknown error'}`);
        }
        const astData = await response.json();

        displayArea.innerHTML = ''; // Clear loading message
        if (astData.length === 0) {
            displayArea.innerHTML = `<p>No AST data found for this file (it might be empty, a non-code file, or not parsed by the backend for AST).</p>`;
            console.log('[AST Viewer] No AST data returned from API for file:', filePath);
            return;
        }
        console.log('[AST Viewer] AST data received, rendering for file:', filePath);
        renderAST(astData, displayArea); // Pass displayArea as the parentElement
    } catch (error) {
        console.error("[AST Viewer] Failed to load or render AST for file " + filePath + ":", error);
        displayArea.innerHTML = `<p style='color:red;'>Failed to load AST data for ${filePath}:<br>${error.message}</p>`;
    }
};

// renderAST function remains the same, but is now called by initializeASTViewer
// Ensure it's defined in a scope accessible by initializeASTViewer (it is, as a global function here)
function renderAST(nodes, parentElement) {
    nodes.forEach(node => {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = 'ast-node';
        nodeDiv.style.marginLeft = (node.depth * 8) + 'px'; // Reduced indentation
        nodeDiv.style.border = '1px dashed #ddd';
        nodeDiv.style.padding = '5px';
        nodeDiv.style.marginBottom = '5px';
        nodeDiv.style.whiteSpace = 'pre-wrap';
        nodeDiv.style.backgroundColor = '#ffffff'; // Default opaque background (white)
        nodeDiv.style.color = '#333';
        nodeDiv.style.position = 'relative'; // For positioning pills if needed

        // Store metadata in data attributes for tooltip
        nodeDiv.dataset.nodeType = node.type;
        nodeDiv.dataset.nodeName = node.name || 'N/A';
        nodeDiv.dataset.startLine = node.start_line;
        nodeDiv.dataset.endLine = node.end_line;
        nodeDiv.dataset.startColumn = node.start_column || 'N/A';
        nodeDiv.dataset.endColumn = node.end_column || 'N/A';
        nodeDiv.dataset.tokenCount = node.token_count || 'N/A';
        nodeDiv.dataset.dgraphDegree = node.dgraph_degree || 'N/A';
        nodeDiv.dataset.childrenCount = node.children_count;

        const pillsContainer = document.createElement('div');
        pillsContainer.className = 'ast-node-pills-container';
        pillsContainer.style.marginBottom = '4px'; // Space between pills and code

        const nodeTypeColors = {
            'function_definition': '#D6EAF8', // Light Blue
            'class_definition': '#FCF3CF',    // Light Yellow
            'method_declaration': '#D5F5E3', // Light Green
            'call_expression': '#FDEDEC',     // Light Pink/Red
            'identifier': '#F2F3F4',         // Very Light Grey
            'import_statement': '#E8DAEF',   // Light Purple
            'if_statement': '#FEF9E7',       // Light Beige
            'for_statement': '#FEF9E7',
            'while_statement': '#FEF9E7',
            'return_statement': '#EBF5FB',   // Another Light Blue shade
            'default': '#E9E9E9'             // Default pill color
        };

        function createPill(text, type = null) {
            const pill = document.createElement('span');
            pill.textContent = text;
            pill.style.display = 'inline-block';
            pill.style.padding = '2px 6px';
            pill.style.marginRight = '4px';
            pill.style.marginBottom = '2px'; // In case they wrap
            pill.style.fontSize = '0.75em';
            pill.style.borderRadius = '8px'; // More rounded pills

            if (type === 'nodeType') {
                pill.style.backgroundColor = nodeTypeColors[node.type.toLowerCase()] || nodeTypeColors['default'];
                // Basic contrast check - if background is very light, use dark text.
                // This is a simplistic check. A proper luminance calculation would be better.
                const bgColor = pill.style.backgroundColor;
                if (bgColor && (bgColor.includes('F8') || bgColor.includes('CF') || bgColor.includes('E3') || bgColor.includes('EC') || bgColor.includes('F4') || bgColor.includes('E7'))) {
                    pill.style.color = '#333';
                } else {
                    pill.style.color = '#555'; // Default for #E9E9E9 or darker custom colors
                }
            } else {
                pill.style.backgroundColor = '#e9e9e9'; // Default for other pills
                pill.style.color = '#555';
            }
            return pill;
        }

        pillsContainer.appendChild(createPill(`Type: ${node.type}`, 'nodeType'));
        pillsContainer.appendChild(createPill(`Lines: L${node.start_line}-L${node.end_line}`));
        if (node.token_count !== 'N/A') {
            pillsContainer.appendChild(createPill(`Tokens: ${node.token_count}`));
        }
        if (node.children_count > 0) {
            pillsContainer.appendChild(createPill(`Children: ${node.children_count}`));
        }
        // Optionally add node name if it exists and isn't too long, or keep it for tooltip
        if (node.name) {
             // pillsContainer.appendChild(createPill(`Name: ${node.name}`)); // Could make pills too wide
        }
        nodeDiv.appendChild(pillsContainer);

        const codeContent = document.createElement('pre');
        codeContent.className = 'ast-node-code';
        codeContent.textContent = node.code_snippet;
        codeContent.style.margin = '0';
        codeContent.style.padding = '0';
        codeContent.style.backgroundColor = 'transparent'; // Ensure code block itself doesn't have conflicting bg
        nodeDiv.appendChild(codeContent);

        // Tooltip setup - remains the same, but default node background is now white
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
                let tooltipContent = `Type: ${nodeDiv.dataset.nodeType}<br>Name: ${nodeDiv.dataset.nodeName}<br>Lines: L${nodeDiv.dataset.startLine}${nodeDiv.dataset.startColumn !== 'N/A' ? ':' + nodeDiv.dataset.startColumn : ''} - L${nodeDiv.dataset.endLine}${nodeDiv.dataset.endColumn !== 'N/A' ? ':' + nodeDiv.dataset.endColumn : ''}<br>Tokens: ${nodeDiv.dataset.tokenCount}<br>Children: ${nodeDiv.dataset.childrenCount}<br>Degree: ${nodeDiv.dataset.dgraphDegree}`;
                tooltip.innerHTML = tooltipContent;
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
// The MutationObserver logic has been removed.
// window.loadAndRenderAST will be called by Gradio's .then() event chain.

// document.addEventListener('DOMContentLoaded', () => {
    // console.log('[AST Viewer] DOMContentLoaded, no observer to set up with .then() approach.');
// });
