from typing import List, Dict

import fitz
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from models.rag import RAG

class ContextWindowEnhancementRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str, paths: str, retriever_k: int = 2, chunk_size: int = 400, chunk_overlap: int = 200, num_neighbors: int = 1):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.paths = paths
        self.retriever_k = retriever_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_neighbors = num_neighbors

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

        self.vectorstore = self.create_vectorstore()
        self.retriever = self.create_retriever()
        self.chain = self.create_chain()

    def create_vectorstore(self):
        chunks = []

        for file_path in self.paths:  # assuming self.paths is a list of paths
            doc = fitz.open(file_path)
            content = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                content += page.get_text()
            doc.close()

            start = 0
            while start < len(content):
                end = start + self.chunk_size
                chunk_text = content[start:end]
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "index": len(chunks),
                            "source_file": file_path,
                            "text": content
                        }
                    )
                )
                start += self.chunk_size - self.chunk_overlap

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        return vectorstore

    def create_retriever(self):
        chunks_query_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})
        return chunks_query_retriever

    def get_chunk_by_index(self, vectorstore, index: int = 1):
        docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
        for doc in docs:
            if doc.metadata.get("index") == index:
                return doc
        return None

    def retrieve_with_context_overlap(self, prompt: str) -> List[str]:
        relevant_chunks = self.retriever.invoke(input=prompt)
        result_sequences = []

        for chunk in relevant_chunks:
            current_index = chunk.metadata.get("index")
            if current_index is None:
                continue

            start_index = max(0, current_index - self.num_neighbors)
            end_index = current_index + self.num_neighbors + 1

            neighbor_chunks = []

            for i in range(start_index, end_index):
                neighbor_chunk = self.get_chunk_by_index(self.vectorstore, i)
                if neighbor_chunk:
                    neighbor_chunks.append(neighbor_chunk)

            neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

            concatenated_text = neighbor_chunks[0].page_content
            for i in range(1, len(neighbor_chunks)):
                current_chunk = neighbor_chunks[i].page_content
                overlap_start = max(0, len(concatenated_text) - self.chunk_overlap)
                concatenated_text = concatenated_text[:overlap_start] + current_chunk

            result_sequences.append(concatenated_text)

        return result_sequences

    def create_chain(self):
        chain = self.prompt | self.llm
        return chain

    def __call__(self, prompt: str) -> Dict:
        content = self.retrieve_with_context_overlap(prompt)
        result = self.chain.invoke({"question": prompt, "context": content})

        return {
            "query": prompt,
            "context": content,
            "result": result,
        }