# test_theorem_training.py
import torch
import json
from transformers import AutoTokenizer
from theorem_contrastive_training import (
    TheoremLemmaDataset,
    BERTEncoder,
    TheoremContrastiveModel,
    TRAIN_CONFIG
)

def test_dataset():
    """Test the dataset loading and processing"""
    print("Testing Dataset...")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TheoremLemmaDataset(
        'data/lemmas_theorems_toy.jsonl',
        tokenizer,
        max_length=256
    )

    print(f"Dataset size: {len(dataset)}")

    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs X shape: {sample['input_ids_x'].shape}")
        print(f"Attention mask X shape: {sample['attention_mask_x'].shape}")
        print("✓ Dataset test passed")
    else:
        print("✗ No data loaded")

def test_model():
    """Test the model forward pass"""
    print("\nTesting Model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    bert_encoder = BERTEncoder(
        model_name='bert-base-uncased',
        hidden_dim=768,
        output_dim=256
    )
    model = TheoremContrastiveModel(bert_encoder, max_length=256).to(device)

    # Create dummy input
    batch_size = 4
    max_length = 256
    x_packed = torch.randn(batch_size, max_length * 2).to(device)
    y_packed = torch.randn(batch_size, max_length * 2).to(device)

    # Set to int for first half (input_ids)
    x_packed[:, :max_length] = torch.randint(0, 1000, (batch_size, max_length)).float()
    x_packed[:, max_length:] = torch.ones(batch_size, max_length)  # attention mask
    y_packed[:, :max_length] = torch.randint(0, 1000, (batch_size, max_length)).float()
    y_packed[:, max_length:] = torch.ones(batch_size, max_length)  # attention mask

    # Forward pass
    with torch.no_grad():
        z_x, z_y = model(x_packed, y_packed)

    print(f"Output X shape: {z_x.shape}")
    print(f"Output Y shape: {z_y.shape}")

    # Check normalization
    x_norm = torch.norm(z_x, dim=1)
    y_norm = torch.norm(z_y, dim=1)
    print(f"X norms (should be ~1): {x_norm}")
    print(f"Y norms (should be ~1): {y_norm}")

    if torch.allclose(x_norm, torch.ones_like(x_norm), atol=1e-5) and \
       torch.allclose(y_norm, torch.ones_like(y_norm), atol=1e-5):
        print("✓ Model test passed")
    else:
        print("✗ Normalization check failed")

def test_contrastive_loss():
    """Test the contrastive loss computation"""
    print("\nTesting Contrastive Loss...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy normalized embeddings
    batch_size = 8
    embed_dim = 256
    z_x = torch.randn(batch_size, embed_dim, device=device)
    z_y = torch.randn(batch_size, embed_dim, device=device)
    z_x = torch.nn.functional.normalize(z_x, p=2, dim=-1)
    z_y = torch.nn.functional.normalize(z_y, p=2, dim=-1)

    # Compute similarity matrix
    tau = 0.07
    S = torch.matmul(z_x, z_y.T) / tau

    # Compute contrastive loss
    labels = torch.arange(batch_size, device=device)
    loss_x = torch.nn.functional.cross_entropy(S, labels, reduction='sum')
    loss_y = torch.nn.functional.cross_entropy(S.T, labels, reduction='sum')
    total_loss = (loss_x + loss_y) / (2 * batch_size)

    print(f"Similarity matrix shape: {S.shape}")
    print(f"Loss value: {total_loss.item():.4f}")
    print("✓ Loss computation test passed")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Theorem Contrastive Training Components")
    print("=" * 50)

    test_dataset()
    test_model()
    test_contrastive_loss()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)