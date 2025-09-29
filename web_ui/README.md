# Theorem & Lemma Browser - Web UI

A web-based interface for browsing, searching, and finding similar mathematical theorems and lemmas using trained neural models.

## Directory Structure

```
web_ui/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Custom CSS styles
│   └── js/
│       └── app.js        # JavaScript frontend logic
└── tests/
    ├── test_ui.py        # UI functionality tests
    ├── test_similarity.py # Similarity search tests
    └── test_toggle.py    # View toggle tests
```

## Features

### 1. Dataset Management
- Load different dataset sizes (toy, tiny, small, medium, full)
- Display dataset statistics (number of papers, theorems, lemmas)
- Datasets stored in `../data/lemmas_theorems/`

### 2. Paper Browser
- Navigate through papers with sidebar navigation
- View theorems, lemmas, and other mathematical statements
- Full LaTeX rendering using MathJax

### 3. Model-Based Similarity Search
- Load trained models (BERT, Qwen)
- Find similar statements using neural embeddings
- GPU-accelerated inference (automatically detects CUDA)
- Top-1000 similar results ranked by cosine similarity

### 4. View Modes
- **Rendered LaTeX**: Display formatted mathematical equations
- **Source Code**: Show raw LaTeX markup in monospace font
- Toggle between modes while preserving context

### 5. Text Search
- Search across all statements in loaded dataset
- Case-insensitive keyword matching

## Installation

### Prerequisites
```bash
# Required Python packages
pip install flask torch transformers peft tqdm

# For GPU acceleration (optional but recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Starting the Server

From the `web_ui` directory:
```bash
python app.py
```

Or from the project root:
```bash
python web_ui/app.py
```

The server will start at `http://localhost:5000`

### Using the Interface

1. **Load a Dataset**:
   - Select a dataset from the dropdown
   - Click "Load Dataset"
   - Papers will appear in the left sidebar

2. **Browse Papers**:
   - Click on any paper in the sidebar
   - View theorems (blue), lemmas (green), and other statements

3. **Use Model-Based Similarity**:
   - Select a model (BERT or Qwen)
   - Click "Load Model" (uses GPU if available)
   - Click "Find Similar" on any theorem/lemma
   - View ranked similar statements

4. **Toggle View Mode**:
   - Use toggle buttons to switch between rendered/source views
   - Useful for copying LaTeX code or viewing formatting

## API Endpoints

- `GET /` - Main interface
- `GET /api/datasets` - List available datasets
- `GET /api/load_dataset/<name>` - Load specific dataset
- `GET /api/paper/<index>` - Get paper details
- `GET /api/models` - List available models
- `GET /api/load_model/<name>` - Load model for similarity
- `POST /api/find_similar` - Find similar statements
- `POST /api/search` - Text search

## Configuration

### Memory Optimization
- Batch size for embedding computation: 8 (adjustable in `app.py`)
- Automatic GPU memory cleanup after each batch
- Embeddings stored on CPU to conserve GPU memory

### Model Paths
- Models expected in `../outputs/`
- LoRA adapters: `../outputs/{model_name}_lora_adapters/`
- Projection layers: `../outputs/{model_name}_projection.pt`

## Testing

Run tests from the `web_ui` directory:

```bash
# Test UI functionality
python tests/test_ui.py

# Test similarity search
python tests/test_similarity.py

# Test view toggle
python tests/test_toggle.py
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `app.py` (line ~265)
- Clear GPU cache: restart the server
- Use CPU instead: models will automatically fallback

### LaTeX Not Rendering
- Ensure internet connection (MathJax loads from CDN)
- Check browser console for errors
- Refresh page after content loads

### Models Not Loading
- Verify model files exist in `../outputs/`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Monitor server logs for detailed error messages

## Performance

- **GPU (RTX 3090)**: ~14 statements/second for embedding computation
- **Similarity search**: <1 second for 1000 results
- **Dataset loading**: Instant (cached in memory)

## Development

### Debug Mode
The server runs in debug mode by default with auto-reload on code changes.

### Adding New Models
1. Train model using main training pipeline
2. Save to `../outputs/` with naming convention
3. Update model name mapping in `app.py` if needed

### Customization
- Modify `static/css/style.css` for styling
- Update `static/js/app.js` for frontend behavior
- Extend `app.py` for new API endpoints