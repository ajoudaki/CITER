import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from typing import Dict
from tqdm import tqdm

# Set TOKENIZERS_PARALLELISM before any other imports to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
# --- NEW: Import PEFT for LoRA ---
from peft import get_peft_model, LoraConfig, TaskType

# Import the new distributed validation function
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

# ===================================================================
# Model and Training Configuration
# ===================================================================
MODEL_CONFIGS = {
    'bert-base': {
        'model_name': 'bert-base-uncased',
        'model_type': 'cls_pooling',
        'hidden_dim': 768,
        # Common target modules for BERT
        'lora_target_modules': ["query", "key", "value"],
    },
    'qwen-1.5b': {
        'model_name': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
        'model_type': 'last_token_pooling',
        'hidden_dim': 1536,
        # Common target modules for Qwen-like models
        'lora_target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
    }
}

# --- CHOOSE YOUR MODEL HERE ---
MODEL_CHOICE = 'qwen-1.5b'
# ------------------------------

SELECTED_MODEL_CONFIG = MODEL_CONFIGS[MODEL_CHOICE]

TRAIN_CONFIG = {
    'GLOBAL_BATCH_SIZE': 64, # Can be increased with LoRA
    'MICRO_BATCH_SIZE': 6,   # Can be increased with LoRA
    'STREAM_CHUNK_SIZE': 16,
    'TAU': 0.07,
    'LR': 2e-4, # A higher LR is common for LoRA
    'NUM_EPOCHS': 10,
    'MAX_LENGTH': 256,
    'OUTPUT_DIM': 768,
    'DROP_LAST': True,
    # --- NEW: LoRA Configuration ---
    'LORA_CONFIG': {
        'enabled': True,
        'r': 16, # Rank of the LoRA matrices
        'lora_alpha': 32, # A scaling factor
        'lora_dropout': 0.05,
        # target_modules are now fetched from MODEL_CONFIGS to be model-agnostic
        'target_modules': SELECTED_MODEL_CONFIG['lora_target_modules'],
    },
}
# Merge model-specific config into the main training config
TRAIN_CONFIG.update(SELECTED_MODEL_CONFIG)


# --- NEW: Helper function to show LoRA's impact ---
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


# ===================================================================
# Dataset for Theorems and Lemmas (Unchanged)
# ===================================================================
class TheoremLemmaDataset(Dataset):
    """Dataset that loads theorem/lemma pairs from JSONL file"""
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, split: str = 'train', train_ratio: float = 0.8, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = []

        all_papers = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                all_statements = paper.get('lemmas', []) + paper.get('theorems', [])
                if len(all_statements) >= 2:
                    all_papers.append(all_statements)
        
        random.seed(seed)
        random.shuffle(all_papers)
        random.seed()

        n_train = int(len(all_papers) * train_ratio)
        self.data = all_papers[:n_train] if split == 'train' else all_papers[n_train:]
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{split.upper()} set: {len(self.data)} papers")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statements = self.data[idx]
        if self.split == 'eval':
            text_x, text_y = (statements[0], statements[1]) if len(statements) >= 2 else (statements[0], statements[0])
        else: # train
            text_x, text_y = random.sample(statements, 2) if len(statements) >= 2 else (statements[0], statements[0])

        tokens_x = self.tokenizer(text_x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        tokens_y = self.tokenizer(text_y, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids_x': tokens_x['input_ids'].squeeze(0), 'attention_mask_x': tokens_x['attention_mask'].squeeze(0),
            'input_ids_y': tokens_y['input_ids'].squeeze(0), 'attention_mask_y': tokens_y['attention_mask'].squeeze(0)
        }
        
# ===================================================================
# Flexible Models and Wrappers
# ===================================================================
class CLSPoolingEncoder(nn.Module):
    """Encoder that uses the [CLS] token embedding."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(cls_embedding), p=2, dim=-1)

class LastTokenEncoder(nn.Module):
    """Encoder that uses the last non-padding token's hidden state."""
    def __init__(self, model_name: str, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(sequence_lengths), device=sequence_lengths.device)
        last_token_embedding = last_hidden_state[batch_indices, sequence_lengths, :]
        return F.normalize(self.projection(last_token_embedding), p=2, dim=-1)

ENCODER_REGISTRY = {
    'cls_pooling': CLSPoolingEncoder,
    'last_token_pooling': LastTokenEncoder,
}

class EncoderWrapper(nn.Module):
    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.base_encoder = base_encoder
        self.max_length = max_length

    def forward(self, packed_tensor):
        input_ids = packed_tensor[:, :self.max_length].long()
        attention_mask = packed_tensor[:, self.max_length:].long()
        return self.base_encoder(input_ids, attention_mask)

class TheoremContrastiveModel(nn.Module):
    def __init__(self, base_encoder, max_length):
        super().__init__()
        self.encoder_x = EncoderWrapper(base_encoder, max_length)
        self.encoder_y = self.encoder_x

    def forward(self, x_packed, y_packed):
        return self.encoder_x(x_packed), self.encoder_y(y_packed)

# ===================================================================
# Training Function (MODIFIED for LoRA)
# ===================================================================
def train(rank: int = 0, world_size: int = 1, distributed: bool = False):
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if distributed: torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Using model: {TRAIN_CONFIG['model_name']} with {TRAIN_CONFIG['model_type']} strategy.")
        if TRAIN_CONFIG['LORA_CONFIG']['enabled']:
            print("LoRA is ENABLED.")

    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset = TheoremLemmaDataset('data/lemmas_theorems_toy.jsonl', tokenizer, max_length=TRAIN_CONFIG['MAX_LENGTH'], split='train')
    val_dataset = TheoremLemmaDataset('data/lemmas_theorems_toy.jsonl', tokenizer, max_length=TRAIN_CONFIG['MAX_LENGTH'], split='eval')

    batch_size = TRAIN_CONFIG['GLOBAL_BATCH_SIZE'] // world_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=4, pin_memory=True,
                              drop_last=TRAIN_CONFIG['DROP_LAST'])
    
    encoder_class = ENCODER_REGISTRY.get(TRAIN_CONFIG['model_type'])
    base_encoder = encoder_class(
        model_name=TRAIN_CONFIG['model_name'],
        hidden_dim=TRAIN_CONFIG['hidden_dim'],
        output_dim=TRAIN_CONFIG['OUTPUT_DIM']
    )

    # --- NEW: Apply LoRA if enabled ---
    if TRAIN_CONFIG['LORA_CONFIG']['enabled']:
        # Freeze all parameters of the base model
        for param in base_encoder.base_model.parameters():
            param.requires_grad = False
            
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # Appropriate for embedding models
            r=TRAIN_CONFIG['LORA_CONFIG']['r'],
            lora_alpha=TRAIN_CONFIG['LORA_CONFIG']['lora_alpha'],
            lora_dropout=TRAIN_CONFIG['LORA_CONFIG']['lora_dropout'],
            target_modules=TRAIN_CONFIG['LORA_CONFIG']['target_modules'],
        )
        
        # Wrap the base model with LoRA adapters
        base_encoder.base_model = get_peft_model(base_encoder.base_model, peft_config)
    
    # The projection head should always be trainable
    for param in base_encoder.projection.parameters():
        param.requires_grad = True

    model = TheoremContrastiveModel(base_encoder, max_length=TRAIN_CONFIG['MAX_LENGTH']).to(device)

    if rank == 0:
        print_trainable_parameters(model)
        
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=TRAIN_CONFIG['LORA_CONFIG']['enabled'])
        
    # --- MODIFIED: Optimizer only sees trainable parameters ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TRAIN_CONFIG['LR'])

    # --- Validation Data Preparation (Unchanged) ---
    val_x_packed, val_y_packed = None, None
    if distributed:
        val_objects = [None, None]
        if rank == 0:
            print("Preparing and distributing validation dataset...")
            val_data = [val_dataset[i] for i in range(len(val_dataset))]
            val_x = torch.stack([d['input_ids_x'] for d in val_data]); val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
            val_y = torch.stack([d['input_ids_y'] for d in val_data]); val_my = torch.stack([d['attention_mask_y'] for d in val_data])
            val_x_packed = torch.cat([val_x, val_mx], dim=1); val_y_packed = torch.cat([val_y, val_my], dim=1)
            val_objects = [val_x_packed, val_y_packed]
        dist.broadcast_object_list(val_objects, src=0)
        val_x_packed, val_y_packed = val_objects
    else:
        print("Preparing validation dataset...")
        val_data = [val_dataset[i] for i in range(len(val_dataset))]
        val_x = torch.stack([d['input_ids_x'] for d in val_data]); val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
        val_y = torch.stack([d['input_ids_y'] for d in val_data]); val_my = torch.stack([d['attention_mask_y'] for d in val_data])
        val_x_packed = torch.cat([val_x, val_mx], dim=1); val_y_packed = torch.cat([val_y, val_my], dim=1)

    # --- Training and Validation Loops (Unchanged) ---
    for epoch in range(TRAIN_CONFIG['NUM_EPOCHS']):
        model.train()
        if distributed and train_sampler: train_sampler.set_epoch(epoch)
        total_loss, num_batches = 0, 0

        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}")
        for batch in pbar:
            x_packed = torch.cat([batch['input_ids_x'].to(device), batch['attention_mask_x'].to(device)], dim=1)
            y_packed = torch.cat([batch['input_ids_y'].to(device), batch['attention_mask_y'].to(device)], dim=1)
            actual_batch_size = x_packed.shape[0] * (world_size if distributed else 1)

            if TRAIN_CONFIG['DROP_LAST'] and actual_batch_size < TRAIN_CONFIG['GLOBAL_BATCH_SIZE']:
                if rank == 0: print(f"Skipping incomplete batch of size {actual_batch_size} (< {TRAIN_CONFIG['GLOBAL_BATCH_SIZE']})")
                continue

            config_copy = TRAIN_CONFIG.copy()
            config_copy['GLOBAL_BATCH_SIZE'] = actual_batch_size
            loss = distributed_train_step(model, optimizer, x_packed, y_packed, config_copy) if distributed else trivial_contrastive_step(model, optimizer, x_packed, y_packed, config_copy)

            total_loss += loss; num_batches += 1
            if rank == 0: pbar.set_postfix(loss=f'{loss:.4f}')

        # --- VALIDATION PHASE ---
        N_val = val_x_packed.shape[0]; val_world_size = world_size if distributed else 1
        C_val = N_val // val_world_size
        if N_val % val_world_size != 0 and rank == 0: print(f"Warning: Val set size {N_val} not divisible by world size {val_world_size}. Truncating.")
        start, end = rank * C_val, (rank + 1) * C_val
        local_val_x, local_val_y = val_x_packed[start:end].to(device), val_y_packed[start:end].to(device)
        val_config = TRAIN_CONFIG.copy(); val_config['GLOBAL_BATCH_SIZE'] = C_val * val_world_size
        val_loss, topk_acc = distributed_validate_step(model, local_val_x, local_val_y, val_config)
        
        if rank == 0:
            avg_train_loss = total_loss / num_batches
            print(f"\nEpoch [{epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}] Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}"); print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MRR:        {topk_acc.get('MRR', 0):.4f}"); print(f"  Top@1 Acc:  {topk_acc.get(1, 0)*100:.2f}%")
            print(f"  Top@5 Acc:  {topk_acc.get(5, 0)*100:.2f}%"); print(f"  Top@10 Acc: {topk_acc.get(10, 0)*100:.2f}%\n")

        if distributed: dist.barrier()

    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        # --- MODIFIED: Save LoRA adapters correctly ---
        if TRAIN_CONFIG['LORA_CONFIG']['enabled']:
            # Save only the LoRA adapters and the projection head
            model_to_save.encoder_x.base_encoder.base_model.save_pretrained(f'{MODEL_CHOICE}_lora_adapters')
            torch.save(model_to_save.encoder_x.base_encoder.projection.state_dict(), f'{MODEL_CHOICE}_projection.pt')
            print(f"LoRA adapters saved to '{MODEL_CHOICE}_lora_adapters' and projection head to '{MODEL_CHOICE}_projection.pt'")
        else:
            torch.save(model_to_save.state_dict(), f'{MODEL_CHOICE}_contrastive_model.pt')
            print(f"Model saved to {MODEL_CHOICE}_contrastive_model.pt")

# ===================================================================
# Main Entry Point (Unchanged)
# ===================================================================
if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        print(f"Running in distributed mode. Rank: {rank}, World Size: {world_size}")
        train(rank, world_size, distributed=True)
        dist.destroy_process_group()
    else:
        print("Running in single device mode")
        train(distributed=False)