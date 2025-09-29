# Theorem & Lemma Browser UI

A web-based interface for browsing mathematical theorems and lemmas with proper LaTeX rendering.

## Features

- **Dataset Selection**: Load different dataset sizes (toy, tiny, small, medium, full)
- **Paper Browser**: Navigate through papers with index-based selection
- **LaTeX Rendering**: Full MathJax support for mathematical notation
- **Statement Organization**: Separate views for theorems, lemmas, and other statements
- **Search Functionality**: Search across all statements in the loaded dataset
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.7+
- Flask (`pip install flask`)

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Load a Dataset**:
   - Select a dataset from the dropdown (start with 'toy' for testing)
   - Click "Load Dataset"

2. **Browse Papers**:
   - Click on any paper in the left sidebar
   - View theorems and lemmas with rendered LaTeX

3. **Search**:
   - Enter search terms in the search box
   - Click "Search" to find matching statements across all papers

## Interface Components

### Left Sidebar
- Dataset selector
- Paper list with counts of theorems/lemmas
- Search functionality

### Main Content Area
- Paper title and ArXiv ID
- Organized sections for:
  - Theorems (blue badges)
  - Lemmas (green badges)
  - Other statements (gray badges)

## Technical Details

- **Backend**: Flask REST API
- **Frontend**: Vanilla JavaScript with Bootstrap
- **LaTeX Rendering**: MathJax 3
- **Data Format**: JSONL files in `data/lemmas_theorems/`

## API Endpoints

- `GET /api/datasets` - List available datasets
- `GET /api/load_dataset/<name>` - Load a specific dataset
- `GET /api/paper/<index>` - Get paper details
- `POST /api/search` - Search statements

## Troubleshooting

If LaTeX is not rendering:
- Ensure you have an internet connection (MathJax loads from CDN)
- Check browser console for errors
- Try refreshing the page after content loads

## Development

To run in debug mode with auto-reload:
```bash
export FLASK_ENV=development
python app.py
```