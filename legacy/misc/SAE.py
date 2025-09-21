import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


class MemoryCachedTransformerDataset(Dataset):
    """Dataset for extracting and caching transformer layer activations in memory"""
    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        model_name: str = "gpt2",
        layer_idx: int = 6,
        max_length: int = 128,
        max_samples: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"Loading dataset {dataset_name}/{dataset_config}...")
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        print(f"Loading model {model_name}...")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.layer_idx = layer_idx
        self.max_length = max_length
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        
        # Compute and store all activations in memory
        print("Computing and caching activations in memory...")
        self.activations = []
        
        for i in tqdm(range(len(self.dataset))):
            tokens = self.tokenizer(
                self.dataset[i]['text'],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move tokens to device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Extract activations
            with torch.no_grad():
                activations = self._extract_activations(tokens)
            
            self.activations.append(activations)
        
        # Convert to single tensor for more efficient storage and indexing
        self.activations = torch.stack(self.activations)
        
        # Clean up the model and tokenizer since we don't need them anymore
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
    
    def _extract_activations(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract activations from specified layer"""
        activations = []
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
                
            if isinstance(act, torch.Tensor):
                activations.append(act.detach())
            else:
                raise ValueError(f"Unexpected activation type: {type(act)}")
        
        # Get the appropriate layer
        if hasattr(self.model, 'encoder'):
            layer = self.model.encoder.layer[self.layer_idx]
        elif hasattr(self.model, 'h'):
            layer = self.model.h[self.layer_idx]
        elif hasattr(self.model, 'layers'):
            layer = self.model.layers[self.layer_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[self.layer_idx]
        else:
            raise ValueError(f"Unknown model architecture")
        
        # For LLaMA models
        if 'llama' in self.model.config.model_type.lower():
            if hasattr(layer, 'mlp'):
                layer = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                layer = layer.feed_forward
        
        handle = layer.register_forward_hook(hook)
        
        try:
            self.model(**tokens)
            if not activations:
                raise ValueError("No activations captured by hook")
            return activations[0].squeeze(0)  # [seq_len, hidden_size]
        finally:
            handle.remove()
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]
        

class SparseAutoencoder(torch.nn.Module):
    """Sparse autoencoder with enhanced sparsity mechanisms"""
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        sparsity_lambda: float = 0.1,  # Increased from 1e-3
        dict_norm_lambda: float = 1e-3,
        target_sparsity: float = 0.05,  # Target activation rate (5%)
        activation_threshold: float = 1e-6,  # Increased threshold
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=False),
            torch.nn.ReLU(),  # ReLU for positive sparse activations
        )
        
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize weights
        with torch.no_grad():
            # Initialize encoder weights
            encoder_layer = self.encoder[0]  # Get the linear layer
            encoder_init = torch.randn(hidden_dim, input_dim)
            encoder_init = torch.nn.functional.normalize(encoder_init, dim=1)
            encoder_layer.weight.data = encoder_init
            
            # Initialize decoder weights
            decoder_init = torch.randn(input_dim, hidden_dim)
            decoder_init = torch.nn.functional.normalize(decoder_init, dim=0)
            self.decoder.weight.data = decoder_init
            
        self.sparsity_lambda = sparsity_lambda
        self.dict_norm_lambda = dict_norm_lambda
        self.target_sparsity = target_sparsity
        self.activation_threshold = activation_threshold
        self.device = device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h
        
    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x_recon, h = self.forward(x)
        
        # Basic losses
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        
        # L1 sparsity
        l1_loss = torch.mean(torch.abs(h))
        
        # KL divergence sparsity penalty
        rho_hat = torch.mean(h, dim=0)  # Average activation of each hidden unit
        kl_loss = torch.mean(self.kl_divergence_sparsity(rho_hat))
        
        # Dictionary element norm loss
        dict_norm_loss = torch.mean((torch.norm(self.decoder.weight, dim=0) - 1)**2)
        
        # Combined loss with both L1 and KL sparsity penalties
        total_loss = (
            recon_loss + 
            self.sparsity_lambda * (l1_loss + kl_loss) +
            self.dict_norm_lambda * dict_norm_loss
        )
        
        # Compute metrics
        sparsity_ratio = torch.mean((torch.abs(h) > self.activation_threshold).float())
        max_activation = torch.max(torch.abs(h))
        feature_usage = torch.mean((torch.abs(h) > self.activation_threshold).float(), dim=0)
        dead_features = torch.sum(feature_usage == 0).item()
        
        metrics = {
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item(),
            'dict_norm_loss': dict_norm_loss.item(),
            'total_loss': total_loss.item(),
            'sparsity_ratio': sparsity_ratio.item(),
            'max_activation': max_activation.item(),
            'dead_features': dead_features,
            'feature_usage_std': torch.std(feature_usage).item()
        }
        
        return total_loss, metrics

    def compute_metrics(self, x: torch.Tensor) -> dict:
        """
        Compute detailed metrics for sparse autoencoder analysis
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary containing various metrics:
            - Basic loss terms (reconstruction, L1, KL divergence)
            - Sparsity measurements
            - Feature activation statistics
            - Neuron specialization metrics
        """
        x_recon, h = self.forward(x)
        batch_size = x.size(0)
        
        # 1. Basic Losses
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        l1_loss = torch.mean(torch.abs(h))
        
        # 2. KL Divergence Loss for Sparsity
        rho_hat = torch.mean(h, dim=0)  # Average activation of each hidden unit
        kl_loss = torch.mean(self.kl_divergence_sparsity(rho_hat))
        
        # 3. Dictionary Element Norm Loss
        dict_norm_loss = torch.mean((torch.norm(self.decoder.weight, dim=0) - 1)**2)
        
        # 4. Total Loss
        total_loss = (
            recon_loss + 
            self.sparsity_lambda * (l1_loss + kl_loss) +
            self.dict_norm_lambda * dict_norm_loss
        )
        
        # 5. Sparsity Metrics
        # Count activations above threshold
        active_neurons = (torch.abs(h) > self.activation_threshold).float()
        sparsity_ratio = torch.mean(active_neurons)
        
        # Per-neuron activation frequencies
        neuron_activity = torch.mean(active_neurons, dim=0)  # [hidden_dim]
        dead_features = torch.sum(neuron_activity == 0).item()
        rarely_active = torch.sum(neuron_activity < 0.01).item()  # <1% activation rate
        hyperactive = torch.sum(neuron_activity > 0.5).item()  # >50% activation rate
        
        # 6. Activation Statistics
        max_activation = torch.max(torch.abs(h))
        mean_activation = torch.mean(torch.abs(h))
        std_activation = torch.std(torch.abs(h))
        
        # 7. Feature Usage Distribution
        feature_usage_std = torch.std(neuron_activity).item()
        feature_usage_entropy = -torch.sum(
            neuron_activity * torch.log(neuron_activity + 1e-10)
        ).item()
        
        # 8. Neuron Correlation Analysis
        # Compute pairwise correlations between most active neurons
        top_k = min(100, h.size(1))  # Use top 100 neurons or all if less
        _, top_indices = torch.topk(neuron_activity, top_k)
        top_activations = h[:, top_indices]
        correlations = torch.corrcoef(top_activations.T)
        mean_correlation = torch.mean(torch.abs(correlations - torch.eye(top_k, device=correlations.device))).item()
        
        # 9. Reconstruction Quality per Feature
        # Compute how much each feature contributes to reconstruction
        with torch.no_grad():
            feature_importance = []
            for i in range(min(100, h.size(1))):  # Sample 100 features for efficiency
                h_zeroed = h.clone()
                h_zeroed[:, i] = 0
                x_recon_zeroed = self.decoder(h_zeroed)
                feature_importance.append(
                    torch.nn.functional.mse_loss(x_recon_zeroed, x).item()
                )
            feature_importance = torch.tensor(feature_importance)
            feature_importance_std = torch.std(feature_importance).item()
        
        metrics = {
            # Loss components
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item(),
            'dict_norm_loss': dict_norm_loss.item(),
            'total_loss': total_loss.item(),
            
            # Sparsity metrics
            'sparsity_ratio': sparsity_ratio.item(),
            'dead_features': dead_features,
            'rarely_active': rarely_active,
            'hyperactive': hyperactive,
            
            # Activation statistics
            'max_activation': max_activation.item(),
            'mean_activation': mean_activation.item(),
            'std_activation': std_activation.item(),
            
            # Feature usage metrics
            'feature_usage_std': feature_usage_std,
            'feature_usage_entropy': feature_usage_entropy,
            'mean_correlation': mean_correlation,
            'feature_importance_std': feature_importance_std
        }
        
        return metrics
    
    def kl_divergence_sparsity(self, rho_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between average activation and target sparsity
        
        Args:
            rho_hat: Average activation of hidden units [hidden_dim]
            
        Returns:
            KL divergence loss measuring deviation from target sparsity
        """
        epsilon = 1e-10  # Small constant for numerical stability
        rho = self.target_sparsity
        
        # Clip values to prevent log(0)
        rho_hat = torch.clamp(rho_hat, epsilon, 1 - epsilon)
        
        # KL divergence
        kl_div = rho * torch.log((rho + epsilon) / (rho_hat + epsilon)) + \
                 (1 - rho) * torch.log((1 - rho + epsilon) / (1 - rho_hat + epsilon))
        
        return kl_div
    
    def analyze_feature(self, feature_idx: int, dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Analyze a specific feature's behavior across the dataset
        
        Args:
            feature_idx: Index of the feature to analyze
            dataloader: DataLoader containing samples to analyze
            
        Returns:
            Dictionary containing feature analysis
        """
        self.eval()
        activations = []
        coactivations = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.reshape(-1, batch.size(-1)).to(self.device)
                _, h = self.forward(batch)
                
                # Get activations for this feature
                feature_acts = h[:, feature_idx]
                activations.append(feature_acts)
                
                # Get co-activated features
                active_mask = torch.abs(h) > self.activation_threshold
                coactive = active_mask & (active_mask[:, feature_idx].unsqueeze(1))
                coactivations.append(coactive)
        
        activations = torch.cat(activations)
        coactivations = torch.cat(coactivations)
        
        analysis = {
            'mean_activation': torch.mean(activations).item(),
            'std_activation': torch.std(activations).item(),
            'activation_rate': torch.mean((torch.abs(activations) > self.activation_threshold).float()).item(),
            'max_activation': torch.max(torch.abs(activations)).item(),
            'top_coactivated_features': torch.topk(torch.sum(coactivations, dim=0), k=5)[1].tolist()
        }
        
        return analysis

def evaluate_model(
    model: SparseAutoencoder,
    dataloader: DataLoader,
    device: str
) -> dict:
    """Evaluate model on given dataloader"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.reshape(-1, batch.size(-1)).to(device)
            metrics = model.compute_metrics(batch)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    return avg_metrics

def plot_training_history(history: dict):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['total_loss', 'recon_loss', 'sparsity_ratio', 'dead_features']
    
    for ax, metric in zip(axes.flat, metrics):
        ax.plot(history[f'train_{metric}'], label='Train')
        ax.plot(history[f'val_{metric}'], label='Validation')
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_autoencoder(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    model_name: str = "gpt2",
    layer_idx: int = 6,
    hidden_dim: int = 4096,
    batch_size: int = 32,
    num_epochs: int = 50,
    max_samples: Optional[int] = 10000,
    lr: float = 5e-4,  # Reduced learning rate
    sparsity_lambda: float = 0.1,  # Increased sparsity penalty
    target_sparsity: float = 0.05,  # Target 5% activation
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_split: float = 0.1
) -> tuple[SparseAutoencoder, dict]:
    """Train sparse autoencoder with validation"""
    # Create full dataset
    full_dataset = MemoryCachedTransformerDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        model_name=model_name,
        layer_idx=layer_idx,
        max_samples=max_samples,
        device=device
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Initialize model with new parameters
    sae = SparseAutoencoder(
        input_dim=full_dataset.hidden_size,
        hidden_dim=hidden_dim,
        sparsity_lambda=sparsity_lambda,
        target_sparsity=target_sparsity,
        device=device
    )
    
    
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training history
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_sparsity_ratio': [], 'val_sparsity_ratio': [],
        'train_dead_features': [], 'val_dead_features': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        sae.train()
        epoch_metrics = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.reshape(-1, full_dataset.hidden_size).to(device)
            
            loss, metrics = sae.loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_metrics.append(metrics)
        
        train_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        
        # Validation
        val_metrics = evaluate_model(sae, val_loader, device)
        
        # Update history
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['train_sparsity_ratio'].append(train_metrics['sparsity_ratio'])
        history['val_sparsity_ratio'].append(val_metrics['sparsity_ratio'])
        history['train_dead_features'].append(train_metrics['dead_features'])
        history['val_dead_features'].append(val_metrics['dead_features'])
        
        # Print metrics
        print(f"\nEpoch {epoch+1} metrics:")
        print("Train:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
        print("Val:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
        
        # Learning rate scheduling
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(sae.state_dict(), 'best_sae.pt')
        
        # Plot training history
        plot_training_history(history)
    
    return sae, history

if __name__ == "__main__":
    sae, history = train_autoencoder(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        model_name="meta-llama/Llama-3.2-1B",
        max_samples=3000,
        num_epochs=50,
        batch_size=16,
        hidden_dim=5000,
        val_split=0.2,
        sparsity_lambda=100,
    )