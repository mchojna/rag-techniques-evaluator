import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API keys from environment variable or .env file
load_dotenv()

# Configure OpenAI API access
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

# Configure main language model
BASE_MODEL = os.getenv("BASE_MODEL", "text-embedding-ada-002")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gpt-4o-mini")

# Set up default configurations
DEFAULT_CHUNK_SIZE = os.getenv("DEFAULT_CHUNK_SIZE", 1000)
DEFAULT_CHUNK_OVERLAP = os.getenv("BASE_DEFAULT_CHUNK_OVERLAPMODEL", 200)
DEFAULT_RETRIEVER_K = os.getenv("DEFAULT_RETRIEVER_K", 4)

model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
