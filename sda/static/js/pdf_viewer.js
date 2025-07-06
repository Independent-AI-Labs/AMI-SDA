// sda/static/js/pdf_viewer.js
document.addEventListener('DOMContentLoaded', initializePDFViewer);

async function initializePDFViewer() {
    console.log('[PDF Viewer] DOMContentLoaded, initializing PDFViewer...');
    const displayArea = document.getElementById('pdf-visualization-container');

    if (!displayArea) {
        console.error('[PDF Viewer] Container #pdf-visualization-container not found.');
        return;
    }

    const urlParams = new URLSearchParams(window.location.search);
    const pdfDocUuid = urlParams.get('pdf_doc_uuid');

    if (!pdfDocUuid) {
        const errorMsg = "<p class='error-message'>Error: PDF Document UUID (pdf_doc_uuid) missing in iframe URL.</p>";
        console.error("[PDF Viewer] Validation failed: pdf_doc_uuid is missing.");
        displayArea.innerHTML = errorMsg;
        return;
    }

    console.log(`[PDF Viewer] PDF Document UUID: ${pdfDocUuid}`);
    displayArea.innerHTML = `<div class="loading-message">Loading PDF structure for ${pdfDocUuid}...</div>`;

    const apiUrl = `/api/pdf/${pdfDocUuid}/document-ast`;
    console.log("[PDF Viewer] Fetching PDF data from:", apiUrl);

    try {
        const response = await fetch(apiUrl);
        console.log(`[PDF Viewer] API response status: ${response.status}`);

        if (!response.ok) {
            let errorDetail = `API request failed with status ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) {
                console.warn('[PDF Viewer] Could not parse error response as JSON.', e);
            }
            throw new Error(errorDetail);
        }

        const parsedPdfDocument = await response.json();
        console.log("[PDF Viewer] Successfully fetched and parsed PDF data:", parsedPdfDocument);

        displayArea.innerHTML = ''; // Clear loading message
        renderPdfDocument(parsedPdfDocument, displayArea);

        // After rendering, render LaTeX if KaTeX is used
        if (window.renderMathInElement) {
            console.log('[PDF Viewer] Calling renderMathInElement...');
            renderMathInElement(document.body, { // Render on whole body, or specific container
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\(", right: "\\)", display: false},
                    {left: "\\[", right: "\\]", display: true}
                ],
                throwOnError: false
            });
            console.log('[PDF Viewer] renderMathInElement completed.');
        } else {
            console.warn('[PDF Viewer] renderMathInElement not found. KaTeX might not be loaded correctly.');
        }

    } catch (error) {
        console.error(`[PDF Viewer] Failed to load or render PDF ${pdfDocUuid}:`, error);
        displayArea.innerHTML = `<p class='error-message'>Failed to load PDF data for ${pdfDocUuid}:<br>${error.message || 'Unknown error'}</p>`;
    }
}

function renderPdfDocument(parsedPdfDocument, parentElement) {
    if (!parsedPdfDocument || !parsedPdfDocument.pages || parsedPdfDocument.pages.length === 0) {
        parentElement.innerHTML = '<div class="loading-message">No content found in this PDF document.</div>';
        console.log('[PDF Viewer] No pages to render.');
        return;
    }
    console.log(`[PDF Viewer] Rendering ${parsedPdfDocument.pages.length} pages.`);

    parsedPdfDocument.pages.forEach(pageNode => {
        const pageDiv = document.createElement('div');
        pageDiv.className = 'pdf-page';
        pageDiv.dataset.pageNumber = pageNode.page_number;
        pageDiv.id = `pdf-page-${pageNode.page_number}`;
        // console.log(`[PDF Viewer] Rendering page ${pageNode.page_number}`);

        if (pageNode.children && pageNode.children.length > 0) {
            pageNode.children.forEach(elementNode => {
                renderPdfElement(elementNode, pageDiv);
            });
        }
        parentElement.appendChild(pageDiv);
    });
}

function renderPdfElement(elementNode, pageDiv) {
    const elementDiv = document.createElement('div');
    elementDiv.className = 'pdf-element';
    elementDiv.dataset.nodeId = elementNode.id;
    elementDiv.dataset.nodeType = elementNode.type;

    // console.log(`[PDF Viewer] Rendering element type: ${elementNode.type}, ID: ${elementNode.id}`);

    switch (elementNode.type) {
        case 'HEADING':
            const headingLevel = elementNode.metadata?.heading_level || 1;
            const heading = document.createElement(`h${Math.min(Math.max(headingLevel, 1), 6)}`);
            heading.textContent = elementNode.text_content || '';
            elementDiv.appendChild(heading);
            break;
        case 'PARAGRAPH':
        case 'TEXT': // Fallback if PARAGRAPH isn't always used
            const p = document.createElement('p');
            p.className = 'pdf-paragraph';
            p.textContent = elementNode.text_content || '';
            elementDiv.appendChild(p);
            break;
        case 'IMAGE':
            if (elementNode.image_blob_id) {
                const img = document.createElement('img');
                img.src = `/api/pdf/image/${elementNode.image_blob_id}`;
                img.alt = elementNode.metadata?.caption || `Image ${elementNode.id}`;
                img.onerror = () => {
                    console.error(`[PDF Viewer] Failed to load image: ${img.src}`);
                    img.alt = `Failed to load image: ${elementNode.image_blob_id}`;
                };
                elementDiv.appendChild(img);
                if (elementNode.metadata?.caption) {
                    const caption = document.createElement('p');
                    caption.className = 'image-caption';
                    caption.textContent = elementNode.metadata.caption;
                    elementDiv.appendChild(caption);
                }
            } else {
                console.warn(`[PDF Viewer] Image node ${elementNode.id} has no image_blob_id.`);
            }
            break;
        case 'TABLE':
            if (elementNode.metadata?.caption) {
                const captionDisplay = document.createElement('p'); // Or use <caption> if structure allows
                captionDisplay.className = 'table-caption-display';
                captionDisplay.textContent = elementNode.metadata.caption;
                elementDiv.appendChild(captionDisplay);
            }
            // html_content for tables is expected to be a full HTML table structure
            elementDiv.innerHTML += elementNode.html_content || '<table><tr><td>Empty or error in table content</td></tr></table>';
            break;
        case 'FORMULA':
            const formulaDiv = document.createElement('div');
            formulaDiv.className = 'pdf-formula';
            // KaTeX will process this. Ensure LaTeX is properly escaped if it comes from JSON.
            // For auto-render, specific delimiters like $...$ or $$...$$ are usually needed.
            // Here, we assume latex_content is the raw LaTeX string.
            // If it doesn't include delimiters, auto-render might not pick it up unless configured.
            // Wrapping with $$ for display mode:
            formulaDiv.textContent = `$$${elementNode.latex_content || ''}$$`;
            elementDiv.appendChild(formulaDiv);
            break;
        case 'FIGURE_CAPTION': // Example of handling more specific types if available
        case 'TABLE_CAPTION':
        case 'FOOTNOTE':
            const metaP = document.createElement('p');
            metaP.style.fontSize = '0.8em';
            metaP.style.fontStyle = 'italic';
            metaP.textContent = `(${elementNode.type}) ${elementNode.text_content || ''}`;
            elementDiv.appendChild(metaP);
            break;
        case 'UNKNOWN':
        default:
            const unknown = document.createElement('div');
            unknown.style.border = "1px dashed red";
            unknown.style.padding = "5px";
            unknown.textContent = `[Unknown type: ${elementNode.type}] ${elementNode.text_content || elementNode.html_content || 'No textual content'}`;
            elementDiv.appendChild(unknown);
            console.warn(`[PDF Viewer] Encountered unknown element type: ${elementNode.type}`, elementNode);
            break;
    }
    pageDiv.appendChild(elementDiv);
}

// TODO:
// 1. Implement highlighting logic (e.g., on hover/click of .pdf-element).
//    - Add event listeners.
//    - Toggle a 'highlighted' class.
// 2. If this viewer runs in an iframe, consider postMessage for communication if needed.
// 3. Refine styling for all PDFElementTypes.
// 4. Test KaTeX rendering thoroughly. Ensure delimiters in latex_content are handled or added correctly.
//    The current implementation wraps `latex_content` with `$$...$$`.
// 5. Add more robust error handling for image loading.
// 6. Consider how to handle page dimensions/aspect ratios if visual fidelity to original PDF page is important.
//    Currently, it's a flow layout.

console.log('[PDF Viewer] pdf_viewer.js loaded.');
