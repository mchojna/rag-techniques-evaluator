from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from abc import ABC, abstractmethod

class RAG(ABC):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str):
        self.base_model = base_model
        self.embedding_model = embedding_model
        self.open_ai_key = open_ai_key

        self.llm = ChatOpenAI(
            model=self.base_model,
            temperature=0.9,
            api_key=self.open_ai_key,
        )

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.open_ai_key,
        )

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        pass

