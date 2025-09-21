#!/bin/bash

# Preprocessing script to create dataset size variants
# Creates the new structure: data/lemmas_theorems/{tiny,small,medium,full}.jsonl

echo "Creating dataset structure..."

# Create dataset directory if it doesn't exist
mkdir -p data/lemmas_theorems

# Check if source file exists
if [ ! -f "data/lemmas_theorems.jsonl" ]; then
    echo "Error: Source file 'data/lemmas_theorems.jsonl' not found!"
    echo "Please ensure the full dataset is available at data/lemmas_theorems.jsonl"
    exit 1
fi

# Get total line count
TOTAL_LINES=$(wc -l < data/lemmas_theorems.jsonl)
echo "Total papers in dataset: $TOTAL_LINES"

# Define size variants
# Adjust these numbers based on your needs
TINY_SIZE=1000      # ~4MB
SMALL_SIZE=10000    # ~40MB
MEDIUM_SIZE=100000  # ~400MB

echo "Creating dataset variants..."

# Create tiny dataset
echo "  Creating tiny dataset (first $TINY_SIZE papers)..."
head -n $TINY_SIZE data/lemmas_theorems.jsonl > data/lemmas_theorems/tiny.jsonl

# Create small dataset
echo "  Creating small dataset (first $SMALL_SIZE papers)..."
head -n $SMALL_SIZE data/lemmas_theorems.jsonl > data/lemmas_theorems/small.jsonl

# Create medium dataset
echo "  Creating medium dataset (first $MEDIUM_SIZE papers)..."
head -n $MEDIUM_SIZE data/lemmas_theorems.jsonl > data/lemmas_theorems/medium.jsonl

# Copy full dataset
echo "  Copying full dataset..."
cp data/lemmas_theorems.jsonl data/lemmas_theorems/full.jsonl

# Create a toy dataset (optional, for very quick testing)
TOY_SIZE=5000  # ~20MB
if [ $TOTAL_LINES -ge $TOY_SIZE ]; then
    echo "  Creating toy dataset (first $TOY_SIZE papers)..."
    head -n $TOY_SIZE data/lemmas_theorems.jsonl > data/lemmas_theorems/toy.jsonl
fi

# Display created files
echo -e "\nDataset variants created:"
ls -lh data/lemmas_theorems/*.jsonl | awk '{print "  " $NF ": " $5}'

# Create dataset statistics
echo -e "\nDataset statistics:"
for file in data/lemmas_theorems/*.jsonl; do
    filename=$(basename "$file" .jsonl)
    lines=$(wc -l < "$file")
    size=$(ls -lh "$file" | awk '{print $5}')
    echo "  $filename: $lines papers, $size"
done

echo -e "\nPreprocessing complete!"
echo "You can now use these datasets with:"
echo "  python theorem_contrastive_training.py dataset.size=tiny"
echo "  python theorem_contrastive_training.py dataset.size=small"
echo "  python theorem_contrastive_training.py dataset.size=medium"
echo "  python theorem_contrastive_training.py dataset.size=full"