# config.py

# ------------------------------
# General configuration settings
# ------------------------------

# Default verbosity level (0 = silent, 1 = errors only, 2 = info, 3 = debug)
VERBOSE = 2

# Whether to load an existing FAISS index (True) or rebuild it from scratch (False)
LOAD_INDEX = False

# ------------------------------
# Retrieval configuration
# ------------------------------

# Default number of chunks/contexts to retrieve during search
NB_CONTEXTES = 10


# Context window around a given paragraph (number of paragraphs before, number after)
# Example: (1, 2) means include 1 paragraph before and 2 after the selected one
SMALL_TO_BIG = (1, 2)

# ------------------------------
# Embedding model configuration
# ------------------------------

# Hugging Face model name used to compute embeddings
EMBED_MODEL = "BAAI/bge-small-en"

# Dimensionality of the embedding vectors (must match the model's output size)
EMBED_DIM = 384

# Max size of the contexts retrieved
CHUNK_MAX_SIZE = 100

CHUNK_OVERLAP = 5
# ------------------------------
# LLM configuration
# ------------------------------

# Model name or endpoint for the generator (e.g., answer generation)
GENERATOR_MODEL = "llama3:latest"

# Model name for query expansion (e.g., turning short query into detailed one)
EXPANDER_MODEL = "llama3:latest"

# Model name for query rewriting (e.g., rewriting user input for better retrieval)
REWRITER_MODEL = "llama3:latest"


