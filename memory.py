import torch


def save_gpu_stats():

    # Monitor memory usage
    memory_allocated = torch.cuda.memory_allocated()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    max_memory_reserved = torch.cuda.max_memory_reserved()

    # Memory profiling
    memory_stats = torch.cuda.memory_stats()

    # Create a dictionary to store GPU statistics
    gpu_stats = {
        "memory_allocated": memory_allocated,
        "max_memory_allocated": max_memory_allocated,
        "memory_reserved": memory_reserved,
        "max_memory_reserved": max_memory_reserved,
        "memory_stats": memory_stats
    }

    return gpu_stats

