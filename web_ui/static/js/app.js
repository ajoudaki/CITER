// Global variables
let currentDataset = null;
let papers = [];
let currentPaperData = null;
let isSourceView = false;
let currentModel = null;

// Load available datasets on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDatasets();

    // Setup view toggle listeners
    document.getElementById('renderedView').addEventListener('change', function() {
        if (this.checked) {
            isSourceView = false;
            if (currentPaperData) {
                displayPaperContent(currentPaperData, currentPaperData.idx);
            }
        }
    });

    document.getElementById('sourceView').addEventListener('change', function() {
        if (this.checked) {
            isSourceView = true;
            if (currentPaperData) {
                displayPaperContent(currentPaperData, currentPaperData.idx);
            }
        }
    });
});

async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();

        const select = document.getElementById('datasetSelect');
        select.innerHTML = '<option value="">Choose a dataset...</option>';

        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.name;
            option.textContent = `${dataset.name} (${dataset.size})`;
            select.appendChild(option);
        });

        // Auto-select 'toy' dataset if available
        if (datasets.find(d => d.name === 'toy')) {
            select.value = 'toy';
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
        alert('Failed to load datasets');
    }
}

async function loadSelectedDataset() {
    const datasetName = document.getElementById('datasetSelect').value;
    if (!datasetName) {
        alert('Please select a dataset');
        return;
    }

    try {
        const response = await fetch(`/api/load_dataset/${datasetName}`);
        const data = await response.json();

        if (data.success) {
            currentDataset = datasetName;
            papers = data.papers;

            // Update UI
            document.getElementById('paperCount').textContent = data.num_papers;
            document.getElementById('datasetInfo').classList.remove('d-none');
            document.getElementById('modelSelector').classList.remove('d-none');

            // Display paper list
            displayPaperList(papers);

            // Load available models
            loadModels();
        } else {
            alert('Failed to load dataset: ' + data.error);
        }
    } catch (error) {
        console.error('Error loading dataset:', error);
        alert('Failed to load dataset');
    }
}

function displayPaperList(papers) {
    const paperList = document.getElementById('paperList');
    paperList.innerHTML = '';

    papers.forEach((paper, index) => {
        const item = document.createElement('a');
        item.href = '#';
        item.className = 'list-group-item list-group-item-action';
        item.onclick = (e) => {
            e.preventDefault();
            loadPaper(index);
            // Update active state
            document.querySelectorAll('#paperList .list-group-item').forEach(el => {
                el.classList.remove('active');
            });
            item.classList.add('active');
        };

        item.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-1">Paper ${index + 1}</h6>
                <small>${paper.arxiv_id}</small>
            </div>
            <p class="mb-1 small text-truncate">${paper.title}</p>
            <small class="text-muted">
                Theorems: ${paper.num_theorems}, Lemmas: ${paper.num_lemmas}
            </small>
        `;

        paperList.appendChild(item);
    });
}

async function loadPaper(paperIdx) {
    try {
        const response = await fetch(`/api/paper/${paperIdx}`);
        const data = await response.json();

        if (data.success) {
            displayPaperContent(data, paperIdx);
        } else {
            alert('Failed to load paper: ' + data.error);
        }
    } catch (error) {
        console.error('Error loading paper:', error);
        alert('Failed to load paper');
    }
}

function displayPaperContent(paper, paperIdx) {
    // Store the current paper data for view toggling
    currentPaperData = paper;
    currentPaperData.idx = paperIdx;

    // Show the view toggle
    document.getElementById('viewToggleContainer').classList.remove('d-none');

    const content = document.getElementById('paperContent');

    let html = `
        <div class="paper-header mb-4">
            <h2>Paper ${paperIdx + 1}</h2>
            <h4>${paper.title}</h4>
            <p class="text-muted">ArXiv ID: ${paper.arxiv_id}</p>
            <hr>
        </div>
    `;

    // Display Theorems
    if (paper.theorems.length > 0) {
        html += `
            <div class="statements-section mb-4">
                <h3 class="section-title">Theorems (${paper.theorems.length})</h3>
                <div class="statement-list">
        `;

        paper.theorems.forEach((theorem, idx) => {
            html += `
                <div class="statement-card theorem-card">
                    <div class="statement-header">
                        <span class="badge bg-primary">Theorem ${idx + 1}</span>
                        ${currentModel ? `<button class="btn btn-sm btn-outline-primary float-end" onclick="findSimilar(${paperIdx}, ${idx}, 'theorem')">Find Similar</button>` : ''}
                    </div>
                    <div class="statement-content">
                        ${isSourceView ? `<pre class="source-code">${escapeHtml(theorem.text)}</pre>` : theorem.text}
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    // Display Lemmas
    if (paper.lemmas.length > 0) {
        html += `
            <div class="statements-section mb-4">
                <h3 class="section-title">Lemmas (${paper.lemmas.length})</h3>
                <div class="statement-list">
        `;

        paper.lemmas.forEach((lemma, idx) => {
            html += `
                <div class="statement-card lemma-card">
                    <div class="statement-header">
                        <span class="badge bg-success">Lemma ${idx + 1}</span>
                        ${currentModel ? `<button class="btn btn-sm btn-outline-success float-end" onclick="findSimilar(${paperIdx}, ${idx}, 'lemma')">Find Similar</button>` : ''}
                    </div>
                    <div class="statement-content">
                        ${isSourceView ? `<pre class="source-code">${escapeHtml(lemma.text)}</pre>` : lemma.text}
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    // Display Others
    if (paper.others.length > 0) {
        html += `
            <div class="statements-section mb-4">
                <h3 class="section-title">Other Statements (${paper.others.length})</h3>
                <div class="statement-list">
        `;

        paper.others.forEach((stmt, idx) => {
            html += `
                <div class="statement-card other-card">
                    <div class="statement-header">
                        <span class="badge bg-secondary">${stmt.type} ${idx + 1}</span>
                    </div>
                    <div class="statement-content">
                        ${isSourceView ? `<pre class="source-code">${escapeHtml(stmt.text)}</pre>` : stmt.text}
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    content.innerHTML = html;

    // Re-render MathJax for LaTeX only in rendered view
    if (!isSourceView && window.MathJax) {
        MathJax.typesetPromise([content]).then(() => {
            // Math rendering complete
        }).catch((e) => console.error('MathJax error:', e));
    }
}

async function searchStatements() {
    const query = document.getElementById('searchBox').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    if (!currentDataset) {
        alert('Please load a dataset first');
        return;
    }

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        if (data.success) {
            displaySearchResults(data.results);
        } else {
            alert('Search failed: ' + data.error);
        }
    } catch (error) {
        console.error('Error searching:', error);
        alert('Search failed');
    }
}

function displaySearchResults(results) {
    const content = document.getElementById('paperContent');

    let html = `
        <div class="search-results">
            <h3>Search Results (${results.length})</h3>
            <button class="btn btn-sm btn-secondary mb-3" onclick="clearSearch()">Clear Search</button>
            <div class="results-list">
    `;

    if (results.length === 0) {
        html += '<p class="text-muted">No results found</p>';
    } else {
        results.forEach(result => {
            html += `
                <div class="search-result-card">
                    <div class="result-header">
                        <span class="badge bg-info">${result.type}</span>
                        <a href="#" onclick="loadPaper(${result.paper_idx}); return false;">
                            Paper ${result.paper_idx + 1}: ${escapeHtml(result.paper_title)}
                        </a>
                    </div>
                    <div class="result-content">
                        ${escapeHtml(result.text)}
                    </div>
                </div>
            `;
        });
    }

    html += `
            </div>
        </div>
    `;

    content.innerHTML = html;

    // Re-render MathJax
    if (window.MathJax) {
        MathJax.typesetPromise([content]).catch((e) => console.error('MathJax error:', e));
    }
}

function clearSearch() {
    document.getElementById('searchBox').value = '';
    document.getElementById('paperContent').innerHTML = `
        <div class="text-center mt-5">
            <h3>Select a paper from the list</h3>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const models = await response.json();

        const select = document.getElementById('modelSelect');
        select.innerHTML = '<option value="">Choose a model...</option>';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.display;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function loadSelectedModel() {
    const modelName = document.getElementById('modelSelect').value;
    if (!modelName) {
        alert('Please select a model');
        return;
    }

    const button = event.target;
    button.disabled = true;
    button.textContent = 'Loading...';

    try {
        const response = await fetch(`/api/load_model/${modelName}`);
        const data = await response.json();

        if (data.success) {
            currentModel = modelName;
            document.getElementById('modelName').textContent = modelName;
            document.getElementById('modelStatus').classList.remove('d-none');
            button.textContent = 'Model Loaded!';
            setTimeout(() => {
                button.textContent = 'Load Model';
                button.disabled = false;
            }, 2000);
        } else {
            alert('Failed to load model: ' + data.error);
            button.textContent = 'Load Model';
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Failed to load model');
        button.textContent = 'Load Model';
        button.disabled = false;
    }
}

async function findSimilar(paperIdx, stmtIdx, stmtType) {
    if (!currentModel) {
        alert('Please load a model first');
        return;
    }

    try {
        const response = await fetch('/api/find_similar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                paper_idx: paperIdx,
                stmt_idx: stmtIdx,
                stmt_type: stmtType
            })
        });

        const data = await response.json();

        if (data.success) {
            displaySimilarityResults(data);
        } else {
            alert('Failed to find similar statements: ' + data.error);
        }
    } catch (error) {
        console.error('Error finding similar:', error);
        alert('Failed to find similar statements');
    }
}

function displaySimilarityResults(data) {
    const content = document.getElementById('paperContent');

    let html = `
        <div class="similarity-results">
            <h3>Similarity Results</h3>
            <button class="btn btn-sm btn-secondary mb-3" onclick="loadPaper(${data.query.paper_idx})">Back to Paper</button>

            <div class="query-box mb-3">
                <h5>Query Statement (${data.query.type})</h5>
                <div class="statement-content">
                    ${isSourceView ? `<pre class="source-code">${escapeHtml(data.query.text)}</pre>` : data.query.text}
                </div>
            </div>

            <h5>Top Similar Statements (${data.results.length})</h5>
            <div class="results-list">
    `;

    data.results.forEach((result, idx) => {
        html += `
            <div class="similarity-result-card">
                <div class="result-header">
                    <span class="badge bg-info">${result.type}</span>
                    <span class="badge bg-warning">Similarity: ${(result.similarity * 100).toFixed(2)}%</span>
                    <a href="#" onclick="loadPaper(${result.paper_idx}); return false;">
                        Paper ${result.paper_idx + 1}: ${escapeHtml(result.paper_title)}
                    </a>
                </div>
                <div class="result-content" style="max-height: 400px; overflow-y: auto;">
                    ${isSourceView ? `<pre class="source-code">${escapeHtml(result.text)}</pre>` : result.text}
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    content.innerHTML = html;

    // Re-render MathJax if needed
    if (!isSourceView && window.MathJax) {
        MathJax.typesetPromise([content]).catch((e) => console.error('MathJax error:', e));
    }
}