import torch
import torch.distributed as dist
import os
import sys

def main():
    # 1. Initialize Distributed
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"--- Distributed Gather Test ---")
        print(f"World Size: {world_size}")

    # 2. Create Local Tensor (Value = Rank)
    # Each rank creates a tensor [10] filled with its rank number
    local_tensor = torch.full((10,), float(rank), device=device)
    
    # 3. All Gather
    gathered_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_tensor)
    
    # 4. Verify on Rank 0
    if rank == 0:
        result = torch.cat(gathered_list, dim=0)
        print(f"Gathered Shape: {result.shape}")
        
        # Expected: [0,0..., 1,1..., 2,2..., 3,3...]
        print("Checking order...")
        
        is_correct = True
        for i in range(world_size):
            chunk = result[i*10 : (i+1)*10]
            if not torch.all(chunk == float(i)):
                print(f"ERROR: Chunk {i} does not contain value {i}!")
                print(f"Got: {chunk}")
                is_correct = False
                
        if is_correct:
            print("SUCCESS: Gather order is [Rank0, Rank1, Rank2, ...].")
        else:
            print("FAILURE: Gather order is scrambled.")
            
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
