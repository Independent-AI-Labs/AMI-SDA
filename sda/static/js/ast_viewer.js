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
        nodeDiv.className = 'ast-node'; // For potential CSS styling from control_panel.css
        nodeDiv.style.marginLeft = (node.depth * 15) + 'px'; // Indentation based on depth
        nodeDiv.style.border = '1px dashed #ddd';
        nodeDiv.style.padding = '5px';
        nodeDiv.style.marginBottom = '5px';
        nodeDiv.style.whiteSpace = 'pre-wrap'; // Preserve whitespace and newlines in code
        nodeDiv.style.backgroundColor = 'transparent'; // Default background
        nodeDiv.style.color = '#333'; // Default text color for node content if not inherited

        // Store metadata in data attributes for tooltip
        nodeDiv.dataset.nodeType = node.type;
        nodeDiv.dataset.nodeName = node.name || 'N/A';
        nodeDiv.dataset.startLine = node.start_line;
        nodeDiv.dataset.endLine = node.end_line;
        nodeDiv.dataset.startColumn = node.start_column || 'N/A';
        nodeDiv.dataset.endColumn = node.end_column || 'N/A';
        nodeDiv.dataset.tokenCount = node.token_count || 'N/A';
        nodeDiv.dataset.dgraphDegree = node.dgraph_degree || 'N/A'; // Will be 'N/A' for now
        nodeDiv.dataset.childrenCount = node.children_count;

        const header = document.createElement('div');
        header.className = 'ast-node-header'; // For potential CSS styling
        let headerText = `[${node.type}] ${node.name || ''} (L${node.start_line}-L${node.end_line})`;
        if (node.children_count > 0) {
            headerText += ` (Children: ${node.children_count})`;
        }
        header.textContent = headerText;
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
