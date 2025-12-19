"""
Test script for PreprocessedGraphDataset.
Verifies compatibility with StratifiedTheoremDataset interface.
"""

import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from graph_contrastive.dataset import PreprocessedGraphDataset

def test_preprocessed_dataset():
    print("=" * 60)
    print("Testing PreprocessedGraphDataset")
    print("=" * 60)

    # Load tokenizer
    model_name = "bert-base-uncased"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load train dataset
    data_dir = "data/processed"
    print(f"\nLoading train dataset from: {data_dir}")
    train_dataset = PreprocessedGraphDataset(
        data_dir,
        tokenizer,
        max_length=256,
        split='train',
        train_ratio=0.9,
        seed=42,
    )

    # Load eval dataset
    print(f"\nLoading eval dataset from: {data_dir}")
    eval_dataset = PreprocessedGraphDataset(
        data_dir,
        tokenizer,
        max_length=256,
        split='eval',
        train_ratio=0.9,
        seed=42,
    )

    print(f"\n--- Dataset Sizes ---")
    print(f"Train: {len(train_dataset):,} edges")
    print(f"Eval:  {len(eval_dataset):,} edges")

    # Test __getitem__
    print(f"\n--- Testing __getitem__ ---")
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"input_ids_x shape: {sample['input_ids_x'].shape}")
    print(f"attention_mask_x shape: {sample['attention_mask_x'].shape}")
    print(f"input_ids_y shape: {sample['input_ids_y'].shape}")
    print(f"attention_mask_y shape: {sample['attention_mask_y'].shape}")

    # Verify expected keys
    expected_keys = {'input_ids_x', 'attention_mask_x', 'input_ids_y', 'attention_mask_y'}
    assert set(sample.keys()) == expected_keys, f"Unexpected keys: {sample.keys()}"
    print("✓ Output format matches StratifiedTheoremDataset")

    # Test reset_epoch
    print(f"\n--- Testing reset_epoch ---")
    idx_before = train_dataset.shuffled_indices[:5].copy()
    train_dataset.reset_epoch()
    idx_after = train_dataset.shuffled_indices[:5]
    print(f"Indices before reset: {idx_before}")
    print(f"Indices after reset:  {idx_after}")
    assert not all(idx_before == idx_after), "Indices should change after reset_epoch"
    print("✓ reset_epoch shuffles indices")

    # Test a few samples to ensure no errors
    print(f"\n--- Testing batch iteration ---")
    for i in range(5):
        sample = train_dataset[i]
        assert sample['input_ids_x'].shape[0] == 256
        assert sample['input_ids_y'].shape[0] == 256
    print("✓ Successfully retrieved 5 samples")

    # Decode a sample to see actual text
    print(f"\n--- Sample Text (decoded) ---")
    sample = train_dataset[100]
    text_x = tokenizer.decode(sample['input_ids_x'], skip_special_tokens=True)
    text_y = tokenizer.decode(sample['input_ids_y'], skip_special_tokens=True)
    print(f"Source (first 200 chars): {text_x[:200]}...")
    print(f"Target (first 200 chars): {text_y[:200]}...")

    print(f"\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_preprocessed_dataset()
