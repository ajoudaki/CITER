# Citation Matcher

## Overview
This repository contains a deep learning model for matching citations in Wikipedia articles. The system learns to identify and link appropriate citations by understanding the context in which they appear and the content of the cited articles.

## Key Features
- Processes Wikipedia XML dumps efficiently
- Extracts and cleans article content and citations
- Implements a BERT-based neural model for citation matching
- Supports both JSONL and SQLite storage formats
- Memory-efficient training with gradient checkpointing
- Comprehensive evaluation metrics for citation matching

## Project Structure
citation-matcher/
├── data_processing.py
├── modeling.py
├── training.py
└── main.ipynb: main 