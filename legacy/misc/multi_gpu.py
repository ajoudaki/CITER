class CustomDataParallel(nn.Module):
    """Custom DataParallel implementation for more efficient multi-GPU training."""
    
    def __init__(self, model, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        
        # Create a model copy for each GPU
        self.models = nn.ModuleList([
            copy.deepcopy(model).to(f'cuda:{device_id}')
            for device_id in device_ids
        ])
        
        # Sync initial parameters across all models
        self._sync_params()
    
    # def _sync_params(self):
    #     """Synchronize parameters across all model copies."""
    #     for param_list in zip(*[m.parameters() for m in self.models]):
    #         param_data = param_list[0].data
    #         for param in param_list[1:]:
    #             param.data.copy_(param_data)

    def _sync_params(self):
        for param_list in zip(*[m.parameters() for m in self.models]):
            # Average gradients across all GPUs
            mean_param = sum(p.data.to('cpu') for p in param_list) / len(param_list)
            # Update all GPUs with mean
            for param in param_list:
                param.data.copy_(mean_param)
    
    def forward(self, batch_list):
        """
        Forward pass handling multiple batches on multiple GPUs.
        
        Args:
            batch_list: List of batches, one for each GPU
        """
        assert len(batch_list) == len(self.device_ids), \
            f"Number of batches ({len(batch_list)}) must match number of GPUs ({len(self.device_ids)})"
        
        # Process each batch on its corresponding GPU
        outputs = []
        for i, (device_id, batch) in enumerate(zip(self.device_ids, batch_list)):
            # Move batch to appropriate device
            batch = {k: v.to(f'cuda:{device_id}') for k, v in batch.items()}
            # Process batch
            outputs.append(self.models[i](**batch))
        
        return outputs

def train_citation_model(
    model,
    results,
    tokenizer,
    config,
    train_ratio: float = 0.8,
    num_epochs: int = 5,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    device: str = None,
    save_path: str = "citation_model.pt",
    batch_size: int = 128,
    temperatures = [],
    gradient_accumulation_steps: int = 1
):
    # Set up multi-GPU training
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with custom data parallel implementation!")
        model = CustomDataParallel(model)
        devices = [f'cuda:{i}' for i in range(num_gpus)]
        per_gpu_batch_size = batch_size // num_gpus
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
    else:
        devices = ['cuda:0']
        per_gpu_batch_size = batch_size
    
    # Initialize optimizer - one for each GPU model
    optimizers = [
        AdamW(model_copy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for model_copy in (model.models if num_gpus > 1 else [model])
    ]
    
    # Initialize gradient scalers for mixed precision - one per GPU
    scalers = [GradScaler() for _ in range(num_gpus)]
    
    # Enable gradient checkpointing for memory efficiency
    if num_gpus > 1:
        for model_copy in model.models:
            model_copy.transformer.gradient_checkpointing_enable()
    else:
        model.transformer.gradient_checkpointing_enable()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        if epoch < len(temperatures):
            temp = temperatures[epoch]
            if num_gpus > 1:
                for model_copy in model.models:
                    model_copy.config.temperature = temp
            else:
                model.config.temperature = temp
            print(f"Temperature changed to {temp}")
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Create new collated data for this epoch
        print("Collating training data with new random masks...")
        collated = collate(results, tokenizer, config)
        dataset = CitationDataset(collated)
        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]
        
        # Create separate dataloaders for each GPU
        train_dataloaders = [
            DataLoader(
                train_dataset,
                batch_size=per_gpu_batch_size,
                shuffle=True,
                collate_fn=citation_collate_fn,
                num_workers=4
            )
            for _ in range(num_gpus)
        ]
        
        val_dataloaders = [
            DataLoader(
                val_dataset,
                batch_size=per_gpu_batch_size,
                shuffle=False,
                collate_fn=citation_collate_fn,
                num_workers=4
            )
            for _ in range(num_gpus)
        ]
        
        # Training phase
        if num_gpus > 1:
            for model_copy in model.models:
                model_copy.train()
        else:
            model.train()
            
        total_train_loss = 0
        train_steps = 0
        
        # Create iterators for each dataloader
        train_iterators = [iter(loader) for loader in train_dataloaders]
        
        # Calculate number of steps
        total_steps = min(len(loader) for loader in train_dataloaders)
        progress_bar = tqdm.tqdm(range(total_steps), desc="Training")
        
        for step in progress_bar:
            # Reset gradients for all optimizers
            for opt in optimizers:
                opt.zero_grad()
            
            step_loss = 0
            
            # Process a batch on each GPU
            for accumulation_step in range(gradient_accumulation_steps):
                try:
                    # Get batches for all GPUs
                    batches = [next(iterator) for iterator in train_iterators]
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        if num_gpus > 1:
                            outputs = model(batches)
                            # Move losses to CPU before combining
                            losses = [output.loss.detach().cpu() for output in outputs]
                            # Calculate mean loss on CPU
                            batch_loss = torch.stack(losses).mean()
                            # Move mean loss back to GPU for backward pass
                            batch_loss = batch_loss.to(devices[0])
                        else:
                            outputs = model(**batches[0])
                            batch_loss = outputs.loss
                    
                    # Scale loss by accumulation steps
                    batch_loss = batch_loss / gradient_accumulation_steps
                    step_loss += batch_loss.item()
                    
                    # Backward pass with gradient scaling
                    if num_gpus > 1:
                        # Each GPU processes its own backward pass
                        for i, (output, scaler, opt) in enumerate(zip(outputs, scalers, optimizers)):
                            scaled_loss = scaler.scale(output.loss / gradient_accumulation_steps)
                            scaled_loss.backward()
                    else:
                        scalers[0].scale(batch_loss).backward()
                
                except StopIteration:
                    break
            
            # Step optimizers and update scalers
            if num_gpus > 1:
                for scaler, opt in zip(scalers, optimizers):
                    scaler.step(opt)
                    scaler.update()
                # Sync parameters after optimization step
                model._sync_params()
            else:
                scalers[0].step(optimizers[0])
                scalers[0].update()
            
            # Update tracking variables
            total_train_loss += step_loss
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': step_loss})
        
        avg_train_loss = total_train_loss / train_steps
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("Running validation...")
        if num_gpus > 1:
            for model_copy in model.models:
                model_copy.eval()
        else:
            model.eval()
            
        total_val_loss = 0
        val_steps = 0
        
        val_iterators = [iter(loader) for loader in val_dataloaders]
        total_val_steps = min(len(loader) for loader in val_dataloaders)
        progress_bar = tqdm.tqdm(range(total_val_steps), desc="Validation")
        
        with torch.no_grad():
            for step in progress_bar:
                try:
                    # Get batches for all GPUs
                    batches = [next(iterator) for iterator in val_iterators]
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        if num_gpus > 1:
                            outputs = model(batches)
                            # Move losses to CPU before combining
                            losses = [output.loss.detach().cpu() for output in outputs]
                            # Calculate mean loss on CPU
                            batch_loss = torch.stack(losses).mean().item()
                        else:
                            outputs = model(**batches[0])
                            batch_loss = outputs.loss.item()
                    
                    total_val_loss += batch_loss
                    val_steps += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': batch_loss})
                
                except StopIteration:
                    break
        
        avg_val_loss = total_val_loss / val_steps
        print(f"\nValidation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the first model if using multiple GPUs (they're all synced)
            model_to_save = model.models[0] if num_gpus > 1 else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizers[0].state_dict(),
                'scaler_state_dict': scalers[0].state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"Saved new best model to {save_path}")
    
    return model