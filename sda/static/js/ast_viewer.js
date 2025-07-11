// Global state for AST Viewer
window.astViewerGlobalState = {
    currentlyHighlightedNode: null,
    tooltipElement: null,
    displayArea: null, // Will be set in initializeASTViewer
    currentFilePath: null // Store full file path for language detection
};

async function initializeASTViewer() {
    console.log('[AST Viewer] initializeASTViewer called');
    const displayArea = document.getElementById('ast-visualization-container'); // Updated ID from HTML
    window.astViewerGlobalState.displayArea = displayArea;

    if (!displayArea) {
        console.error('[AST Viewer] Container #ast-visualization-container not found.');
        return;
    }

    // Extract parameters from URL query string (iframe source)
    const urlParams = new URLSearchParams(window.location.search);
    const repoId = urlParams.get('repo_id');
    const branchName = urlParams.get('branch_name');
    const filePath = urlParams.get('file_path');
    window.astViewerGlobalState.currentFilePath = filePath; // Store for language detection

    console.log(`[AST Viewer] Params from URL: repoId=${repoId}, branchName=${branchName}, filePath=${filePath}`);

    if (!repoId || !branchName || !filePath) {
        let errorMsg = "<p class='error-message'>Error: Missing parameters in iframe URL.<ul>";
        if (!repoId) errorMsg += "<li>Repository ID (repo_id) missing</li>";
        if (!branchName) errorMsg += "<li>Branch name (branch_name) missing</li>";
        if (!filePath) errorMsg += "<li>File path (file_path) missing</li>";
        errorMsg += "</ul></p>";
        console.error("[AST Viewer] Validation failed:", errorMsg);
        displayArea.innerHTML = errorMsg;
        return;
    }

    displayArea.innerHTML = `<div class="loading-message">Loading AST for ${filePath}...</div>`;

    // Tooltip setup (once)
    if (!window.astViewerGlobalState.tooltipElement) {
        let tooltip = document.createElement('div');
        tooltip.className = 'ast-tooltip'; // Styled by ast_visualization.html CSS
        document.body.appendChild(tooltip);
        window.astViewerGlobalState.tooltipElement = tooltip;
    }

    // Event Handlers for delegation (setup once)
    if (!window.astViewerGlobalState.eventListenersAttached) {
        displayArea.addEventListener('mouseover', handleMouseOver);
        displayArea.addEventListener('mouseout', handleMouseOut);
        // Use mousemove on displayArea to avoid body listeners if iframe is small,
        // but body might be better if tooltip can go outside displayArea bounds.
        // Let's stick to displayArea for now, assuming it's large enough.
        displayArea.addEventListener('mousemove', handleMouseMove);
        window.astViewerGlobalState.eventListenersAttached = true;
    }

    // API call
    const apiUrl = `/api/repo/${repoId}/branch/${encodeURIComponent(branchName)}/file-ast?path=${encodeURIComponent(filePath)}`;
    console.log("[AST Viewer] Fetching AST data from:", apiUrl);

    try {
        const response = await fetch(apiUrl);
        console.log(`[AST Viewer] API response status: ${response.status}`);

        if (!response.ok) {
            let errorDetail = `API request failed with status ${response.status}: ${response.statusText}`;
            try {
                const errorText = await response.text(); // Try to get text first, might not be JSON
                console.error(`[AST Viewer] API error response text: ${errorText}`);
                // Attempt to parse as JSON, but gracefully handle if it's not
                try {
                    const errorData = JSON.parse(errorText);
                    errorDetail = errorData.detail || errorDetail;
                } catch (parseError) {
                    // If JSON parsing fails, use the raw text if it's not too long, or a generic message
                    errorDetail = errorText.substring(0, 200) || errorDetail;
                }
            } catch (e) {
                console.error('[AST Viewer] Could not get error response text/JSON.', e);
            }
            throw new Error(errorDetail);
        }

        const astData = await response.json();
        console.log("[AST Viewer] Successfully fetched and parsed AST data from API.");

        displayArea.innerHTML = ''; // Clear loading message
        if (!astData || astData.length === 0) {
            console.log("[AST Viewer] AST data is empty. Displaying 'No AST data found' message.");
            displayArea.innerHTML = `<div class="loading-message">No AST data found for this file (it might be empty, a non-code file, or not parsed).</div>`;
            return;
        }

        console.log(`[AST Viewer] Rendering ${astData.length} root AST nodes.`);
        renderAST(astData, displayArea);

        // After all nodes are rendered, tell Prism to highlight everything
        if (window.Prism) {
            console.log('[AST Viewer] Calling Prism.highlightAll()');
            Prism.highlightAll();
            console.log('[AST Viewer] Prism.highlightAll() completed.');
        } else {
            console.warn('[AST Viewer] Prism.js not found. Syntax highlighting will not be applied.');
        }
        console.log("[AST Viewer] AST rendering and highlighting complete.");

    } catch (error) {
        console.error(`[AST Viewer] Failed to load or render AST for ${filePath}:`, error);
        displayArea.innerHTML = `<p class='error-message'>Failed to load AST data for ${filePath}:<br>${error.message || 'Unknown error'}</p>`;
    }
}

function renderAST(nodes, parentElement, parentNodeObject = null) {
    const filePath = window.astViewerGlobalState.currentFilePath;
    const extension = filePath ? filePath.split('.').pop().toLowerCase() : '';
    const langMap = {
        'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'java': 'java',
        'c': 'c', 'cpp': 'cpp', 'cs': 'csharp', 'go': 'go', 'rb': 'ruby',
        'php': 'php', 'swift': 'swift', 'kt': 'kotlin', 'rs': 'rust',
        'html': 'markup', 'xml': 'markup', 'svg': 'markup', 'css': 'css',
        'scss': 'scss', 'less': 'less', 'json': 'json', 'yaml': 'yaml',
        'yml': 'yaml', 'md': 'markdown', 'sh': 'bash', 'sql': 'sql',
        'dockerfile': 'docker', 'jsx': 'jsx', 'tsx': 'tsx'
    };
    const languageClass = langMap[extension] || 'clike';

    nodes.forEach(node => {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = 'ast-node';
        // Apply margin if it's a top-level node or if its parent is not directly its container
        // This logic might need refinement based on where renderAST is called from.
        // For true nesting, child nodes are appended to parent's code container,
        // so their depth-based margin should still work relative to the overall structure.
        nodeDiv.style.marginLeft = (node.depth * 15) + 'px';

        nodeDiv.dataset.nodeType = node.type;
        nodeDiv.dataset.nodeName = node.name || 'N/A';
        nodeDiv.dataset.startLine = node.start_line;
        nodeDiv.dataset.endLine = node.end_line;
        nodeDiv.dataset.startColumn = node.start_column || 'N/A';
        nodeDiv.dataset.endColumn = node.end_column || 'N/A';
        nodeDiv.dataset.tokenCount = node.token_count || 'N/A';
        nodeDiv.dataset.dgraphDegree = node.dgraph_degree || 'N/A';
        nodeDiv.dataset.childrenCount = node.children ? node.children.length : 0;

        const header = document.createElement('div');
        header.className = 'ast-node-header';
        header.innerHTML = `
            <span class="node-type">${node.type}</span>
            ${node.name ? `<span class="node-name">${node.name}</span>` : ''}
            <span class="node-lines">(L${node.start_line}-${node.end_line})</span>`;
        nodeDiv.appendChild(header);

        const codeContainer = document.createElement('div');
        codeContainer.className = 'ast-node-code-container';

        if (node.code_snippet) {
            if (!node.children || node.children.length === 0) {
                // Node has no children, render its code snippet directly
                const pre = document.createElement('pre');
                pre.className = `language-${languageClass}`;
                const code = document.createElement('code');
                code.className = `language-${languageClass}`;
                code.textContent = node.code_snippet;
                pre.appendChild(code);
                codeContainer.appendChild(pre);
            } else {
                // Node has children, process snippet line by line
                const lines = node.code_snippet.split('\n'); // Corrected split character
                let currentLineInSnippet = 0;
                let childIndex = 0;
                let lineBuffer = []; // Use an array to store lines for current segment

                const flushLineBuffer = () => {
                    if (lineBuffer.length > 0) {
                        const pre = document.createElement('pre');
                        pre.className = `language-${languageClass}`;
                        const code = document.createElement('code');
                        code.className = `language-${languageClass}`;
                        code.textContent = lineBuffer.join('\n'); // Join lines with a single newline
                        pre.appendChild(code);
                        codeContainer.appendChild(pre);
                        lineBuffer = []; // Reset buffer
                    }
                };

                while (currentLineInSnippet < lines.length) {
                    const currentAbsoluteLineNo = node.start_line + currentLineInSnippet;
                    let childToRender = null;

                    if (childIndex < node.children.length) {
                        const child = node.children[childIndex];
                        if (child.start_line === currentAbsoluteLineNo) {
                            childToRender = child;
                        }
                    }

                    if (childToRender) {
                        flushLineBuffer(); // Render any preceding lines of the parent
                        // Recursively render the child node into the current codeContainer
                        renderAST([childToRender], codeContainer, node); // Pass node as parentNodeObject
                        // Advance snippet lines past this child's content
                        currentLineInSnippet += (childToRender.end_line - childToRender.start_line + 1);
                        childIndex++;
                    } else {
                        // This line is part of the parent node, not a child.
                        lineBuffer.push(lines[currentLineInSnippet]); // Add line to buffer
                        currentLineInSnippet++;
                    }
                }
                flushLineBuffer(); // Render any remaining lines in the buffer
            }
        }

        nodeDiv.appendChild(codeContainer);
        parentElement.appendChild(nodeDiv);

        // The recursive call for children is now handled inside the parent's code rendering logic
        // if (node.children && node.children.length > 0) {
        //    renderAST(node.children, nodeDiv); // OLD RECURSIVE CALL
        // }
    });
}

// Event Handlers
function handleMouseOver(event) {
    const targetNode = event.target.closest('.ast-node');
    let state = window.astViewerGlobalState;

    if (state.currentlyHighlightedNode && state.currentlyHighlightedNode !== targetNode) {
        state.currentlyHighlightedNode.style.backgroundColor = ''; // Revert previous
    }

    if (targetNode) {
        targetNode.style.backgroundColor = '#e0e0e0'; // Highlight color
        state.currentlyHighlightedNode = targetNode;

        let tooltipContent = `Type: ${targetNode.dataset.nodeType}<br>Name: ${targetNode.dataset.nodeName}<br>Lines: L${targetNode.dataset.startLine}${targetNode.dataset.startColumn !== 'N/A' ? ':' + targetNode.dataset.startColumn : ''} - L${targetNode.dataset.endLine}${targetNode.dataset.endColumn !== 'N/A' ? ':' + targetNode.dataset.endColumn : ''}<br>Tokens: ${targetNode.dataset.tokenCount}<br>Children: ${targetNode.dataset.childrenCount}<br>Degree: ${targetNode.dataset.dgraphDegree}`;
        state.tooltipElement.innerHTML = tooltipContent;
        state.tooltipElement.style.visibility = 'visible';
    }
}

function handleMouseOut(event) {
    const targetNode = event.target.closest('.ast-node');
    let state = window.astViewerGlobalState;

    if (targetNode && state.currentlyHighlightedNode === targetNode) {
         // Check if mouse is moving to a child or related element before hiding
        if (!event.relatedTarget || !targetNode.contains(event.relatedTarget)) {
            targetNode.style.backgroundColor = ''; // Revert
            state.currentlyHighlightedNode = null;
            state.tooltipElement.style.visibility = 'hidden';
        }
    } else if (!targetNode && state.currentlyHighlightedNode) {
        // Mouse left the display area or went to a non-node element
        state.currentlyHighlightedNode.style.backgroundColor = '';
        state.currentlyHighlightedNode = null;
        state.tooltipElement.style.visibility = 'hidden';
    }
}

function handleMouseMove(event) {
    let state = window.astViewerGlobalState;
    if (state.tooltipElement && state.tooltipElement.style.visibility === 'visible') {
        // Position tooltip relative to the viewport
        // Add scroll offsets of the iframe's document if the tooltip is appended to iframe's body
        const scrollX = window.scrollX || document.documentElement.scrollLeft;
        const scrollY = window.scrollY || document.documentElement.scrollTop;
        state.tooltipElement.style.left = (event.clientX + scrollX + 15) + 'px';
        state.tooltipElement.style.top = (event.clientY + scrollY + 15) + 'px';
    }
}

// Initialize when the script is loaded and DOM is ready for an iframe.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeASTViewer);
} else {
    initializeASTViewer();
}

console.log('[AST Viewer] ast_viewer.js loaded and ready.');
