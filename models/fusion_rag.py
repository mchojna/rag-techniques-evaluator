from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from pydantic import SecretStr
import numpy as np

from models.rag import RAG


class FusionRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str, paths: str, retriever_k: int = 2,
                 chunk_size: int = 1000, chunk_overlap: int = 0, alpha: float = 0.5):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.paths = paths
        self.retriever_k = retriever_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.alpha = alpha

        self.vectorstore, self.texts = self.create_vectorstore()
        self.bm25 = self.create_bm25_index()

        self.prompt_template = """
            You are an AI assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Question: {question}

            Context: {context}

            Answer:
        """

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
        )

        self.chain = self.create_chain()

    def create_vectorstore(self):
        all_documents = []

        for path in self.paths:
            loader = PyPDFLoader(path)
            documents = loader.load()
            all_documents.extend(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        texts = text_splitter.split_documents(all_documents)

        vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

        return vectorstore, texts

    def create_bm25_index(self) -> BM25Okapi:
        tokenized_docs = [doc.page_content.split() for doc in self.texts]
        return BM25Okapi(tokenized_docs)

    def retrieve_context(self, prompt: str) -> List[Document]:
        epsilon = 1e-8

        all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
        bm25_scores = self.bm25.get_scores(prompt.split())

        vector_result = self.vectorstore.similarity_search_with_score(prompt, k=len(all_docs))

        vector_scores = np.array([score for _, score in vector_result])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (
                np.max(vector_scores) - np.min(vector_scores) + epsilon)

        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

        combined_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores

        sorted_indices = np.argsort(combined_scores)[::-1]
        return [all_docs[i] for i in sorted_indices[:self.retriever_k]]

    def create_chain(self):
        chain = self.prompt | self.llm
        return chain

    def __call__(self, prompt: str) -> str:
        context = self.retrieve_context(prompt)
        result = self.chain.invoke({"question": prompt, "context": context})
        return result