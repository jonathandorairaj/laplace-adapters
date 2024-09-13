import torch


def save_gpu_stats():

    # Monitor memory usage
    """
    Retrieves and returns various GPU memory statistics using PyTorch.

    This function monitors and captures the current and maximum memory usage
    statistics for a CUDA-enabled GPU using PyTorch's `torch.cuda` module.
    

    Returns:
        dict: A dictionary containing the following GPU memory statistics:
            - "memory_allocated": Current memory allocated on the GPU (in bytes).
            - "max_memory_allocated": Maximum memory allocated on the GPU since the beginning (in bytes).
            - "memory_reserved": Current memory reserved by the caching allocator (in bytes).
            - "max_memory_reserved": Maximum memory reserved by the caching allocator since the beginning (in bytes).
            - "memory_stats": A detailed memory statistics dictionary provided by `torch.cuda.memory_stats()`.

    Examples:
        >>> gpu_stats = save_gpu_stats()
        >>> print(gpu_stats["memory_allocated"])
    """
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

