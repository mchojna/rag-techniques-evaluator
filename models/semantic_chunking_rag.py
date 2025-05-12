import fitz
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import SecretStr
from langchain_experimental.text_splitter import SemanticChunker

from rag import RAG

class SemanticChunkingRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: SecretStr, path: str, retriever_k: int = 2, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.path = path
        self.retriever_k = retriever_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        doc = fitz.open(self.path)
        content = ""
        for pag_num in range(len(doc)):
            page = doc[pag_num]
            content += page.get_text()

        text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=90
        )

        chunks = text_splitter.split_text(content)

        documents = [Document(page_content=chunk) for chunk in chunks]

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        return vectorstore

    def create_retriever(self):
        chunks_query_retriever = self.vectorstore.as_retriever(search_k=self.retriever_k)
        return chunks_query_retriever

    def create_chain(self):
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        return chain

    def __call__(self, prompt: str) -> str:
        result = self.chain.invoke(prompt)
        return result["result"]