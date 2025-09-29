#!/bin/bash

# Launch the Theorem & Lemma Browser Web UI

echo "Starting Theorem & Lemma Browser..."
echo "Server will be available at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Navigate to the web_ui directory if not already there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start the Flask server
python app.py