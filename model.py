import os
from dotenv import load_dotenv

# Load API keys from environment variable or .env file
load_dotenv()

# Configure OpenAI API access
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

# Configure main language model
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Set up default configurations
DEFAULT_CHUNK_SIZE = os.getenv("DEFAULT_CHUNK_SIZE", 1000)
DEFAULT_CHUNK_OVERLAP = os.getenv("DEFAULT_CHUNK_OVERLAP", 200)
DEFAULT_RETRIEVER_K = int(os.getenv("DEFAULT_RETRIEVER_K", 4))

# 1. simple_rag vs adaptive_retriever
# 2. semantic_chunking vs context_enrichment_window_around_chunk
# 3. contextual_chunk_headers vs crag

# simple_rag

# adaptive_retriever

# semantic_chunking

# context_enrichment_window_around_chunk

# contextual_chunk_headers

# crag
