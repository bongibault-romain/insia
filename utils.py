# memory_store.py

import json
import os

MEMORY_FILE = "memory.txt"

def save_entry(entry: dict, file_path: str = MEMORY_FILE):
    """Save a dictionary entry as a JSON line in the memory file."""
    with open(file_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_entries(file_path: str = MEMORY_FILE):
    """Load all dictionary entries from the memory file."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def clear_memory(file_path: str = MEMORY_FILE):
    """Clear the memory file."""
    open(file_path, "w").close()

# Example usage:
if __name__ == "__main__":
    # Example entries
    save_entry({'activation': 'relu', 'max_iter': '500', 'validation_score': 0.87})
    save_entry({'activation': 'tanh', 'validation_score': 0.90})
    
    entries = load_entries()
    print("Stored entries:")
    for entry in entries:
        print(entry)
