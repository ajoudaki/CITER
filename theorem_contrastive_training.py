import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Set TOKENIZERS_PARALLELISM before any other imports to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
import random
from typing import Dict
from tqdm import tqdm

# Import the new distributed validation function
from distributed_clip import distributed_train_step, trivial_contrastive_step, distributed_validate_step

# ===================================================================
# Dataset for Theorems and Lemmas
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
# Models and Wrappers
# ===================================================================
class BERTEncoder(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', hidden_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(cls_embedding), p=2, dim=-1)

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
    def __init__(self, bert_encoder, max_length=512):
        super().__init__()
        self.encoder_x = EncoderWrapper(bert_encoder, max_length)
        self.encoder_y = self.encoder_x # Share parameters
    def forward(self, x_packed, y_packed):
        return self.encoder_x(x_packed), self.encoder_y(y_packed)

# ===================================================================
# Training Configuration
# ===================================================================
TRAIN_CONFIG = {
    'GLOBAL_BATCH_SIZE': 1024,
    'MICRO_BATCH_SIZE': 64,
    'STREAM_CHUNK_SIZE': 64,
    'TAU': 0.07,
    'LR': 0.00015,
    'NUM_EPOCHS': 10,
    'MAX_LENGTH': 256,
    'OUTPUT_DIM': 768,
}

# ===================================================================
# Training Function
# ===================================================================
def train(rank: int = 0, world_size: int = 1, distributed: bool = False):
    """Main training function"""
    device = torch.device(f'cuda:{rank}') if distributed and torch.cuda.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if distributed: torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TheoremLemmaDataset('data/lemmas_theorems_toy.jsonl', tokenizer, max_length=TRAIN_CONFIG['MAX_LENGTH'], split='train')
    val_dataset = TheoremLemmaDataset('data/lemmas_theorems_toy.jsonl', tokenizer, max_length=TRAIN_CONFIG['MAX_LENGTH'], split='eval')

    batch_size = TRAIN_CONFIG['GLOBAL_BATCH_SIZE'] // world_size if distributed else TRAIN_CONFIG['GLOBAL_BATCH_SIZE']
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    
    bert_encoder = BERTEncoder(output_dim=TRAIN_CONFIG['OUTPUT_DIM'])
    model = TheoremContrastiveModel(bert_encoder, max_length=TRAIN_CONFIG['MAX_LENGTH']).to(device)
    if distributed: 
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['LR'])

    # --- Prepare and Distribute Validation Data ONCE ---
    val_x_packed, val_y_packed = None, None
    if distributed:
        val_objects = [None, None]
        if rank == 0:
            print("Preparing and distributing validation dataset...")
            val_data = [val_dataset[i] for i in range(len(val_dataset))]
            val_x = torch.stack([d['input_ids_x'] for d in val_data])
            val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
            val_y = torch.stack([d['input_ids_y'] for d in val_data])
            val_my = torch.stack([d['attention_mask_y'] for d in val_data])
            val_x_packed = torch.cat([val_x, val_mx], dim=1)
            val_y_packed = torch.cat([val_y, val_my], dim=1)
            val_objects = [val_x_packed, val_y_packed]
        dist.broadcast_object_list(val_objects, src=0)
        val_x_packed, val_y_packed = val_objects
    else: # Single device case
        print("Preparing validation dataset...")
        val_data = [val_dataset[i] for i in range(len(val_dataset))]
        val_x = torch.stack([d['input_ids_x'] for d in val_data])
        val_mx = torch.stack([d['attention_mask_x'] for d in val_data])
        val_y = torch.stack([d['input_ids_y'] for d in val_data])
        val_my = torch.stack([d['attention_mask_y'] for d in val_data])
        val_x_packed = torch.cat([val_x, val_mx], dim=1)
        val_y_packed = torch.cat([val_y, val_my], dim=1)


    for epoch in range(TRAIN_CONFIG['NUM_EPOCHS']):
        model.train()
        if distributed and train_sampler: train_sampler.set_epoch(epoch)
        total_loss, num_batches = 0, 0

        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}")
        for batch in pbar:
            x_packed = torch.cat([batch['input_ids_x'].to(device), batch['attention_mask_x'].to(device)], dim=1)
            y_packed = torch.cat([batch['input_ids_y'].to(device), batch['attention_mask_y'].to(device)], dim=1)

            actual_batch_size = x_packed.shape[0] * (world_size if distributed else 1)
            config_copy = TRAIN_CONFIG.copy()
            config_copy['GLOBAL_BATCH_SIZE'] = actual_batch_size
            
            loss = distributed_train_step(model, optimizer, x_packed, y_packed, config_copy) if distributed else trivial_contrastive_step(model, optimizer, x_packed, y_packed, config_copy)
            
            total_loss += loss
            num_batches += 1
            if rank == 0: pbar.set_postfix(loss=f'{loss:.4f}')

        # --- VALIDATION PHASE (RUNS ON ALL GPUs) ---
        N_val = val_x_packed.shape[0]
        val_world_size = world_size if distributed else 1
        C_val = N_val // val_world_size
        
        if N_val % val_world_size != 0 and rank == 0:
            print(f"Warning: Val set size {N_val} not divisible by world size {val_world_size}. Truncating.")

        start, end = rank * C_val, (rank + 1) * C_val
        local_val_x = val_x_packed[start:end].to(device)
        local_val_y = val_y_packed[start:end].to(device)

        val_config = TRAIN_CONFIG.copy()
        val_config['GLOBAL_BATCH_SIZE'] = C_val * val_world_size
        
        val_loss, topk_acc = distributed_validate_step(model, local_val_x, local_val_y, val_config)
        
        if rank == 0:
            avg_train_loss = total_loss / num_batches
            print(f"\nEpoch [{epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}] Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Top@1 Acc:  {topk_acc.get(1, 0)*100:.2f}%")
            print(f"  Top@5 Acc:  {topk_acc.get(5, 0)*100:.2f}%")
            print(f"  Top@10 Acc: {topk_acc.get(10, 0)*100:.2f}%\n")

        # Add a barrier to ensure all processes sync up before the next training epoch
        if distributed:
            dist.barrier()

    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 'theorem_contrastive_model.pt')
        print("Model saved to theorem_contrastive_model.pt")

# ===================================================================
# Main Entry Point
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


