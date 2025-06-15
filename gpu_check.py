import shutil
import torch

print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("\nNumber of GPUs:", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} Info ---")
        props = torch.cuda.get_device_properties(i)
        
        print("Name:", props.name)
        print("Compute Capability:", f"{props.major}.{props.minor}")
        print("Total Memory (MB):", round(props.total_memory / 1024 ** 2, 2))
        print("Multi-Processor Count:", props.multi_processor_count)
        print("CUDA Cores (approx):", props.multi_processor_count * 64)  # Assumes 64 cores/SM

        print("Memory Allocated (MB):", round(torch.cuda.memory_allocated(i) / 1024 ** 2, 2))
        print("Memory Reserved (MB):", round(torch.cuda.memory_reserved(i) / 1024 ** 2, 2))

        # additional attributes which don't seem to exist in all envs
        try:
            print("Max Threads per Block:", props.max_threads_per_block)
            print("Max Threads per SM:", props.max_threads_per_multiprocessor)
            print("Max Grid Size:", props.max_grid_size)
            print("Max Threads Dim:", props.max_threads_dim)
        except AttributeError:
            print("Some detailed fields (e.g., max threads) not available in this PyTorch version.")

print('\n')

if shutil.which("nvidia-smi"):
    !nvidia-smi
else:
    print("nvidia-smi not available (no NVIDIA GPU or drivers).")

print("\n--- CPU Info ---")
print("Number of Cores:", os.cpu_count())
