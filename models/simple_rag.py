from typing import Dict

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.rag import RAG

class SimpleRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str, paths: str, retriever_k: int = 2, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.paths = paths
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
        all_documents = []

        for file_path in self.paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(all_documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)

        return vectorstore

    def create_retriever(self):
        chunks_query_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})
        return chunks_query_retriever

    def create_chain(self):
        # chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.retriever,
        #     chain_type_kwargs={"prompt": self.prompt},
        #     return_source_documents=True
        # )
        # return chain
        chain = self.prompt | self.llm
        return chain

    def __call__(self, prompt: str) -> Dict:
        # result = self.chain.invoke(prompt)
        # return result
        content = self.retriever.invoke(input=prompt)
        result = self.chain.invoke({"question": prompt, "context": content})

        return {
            "query": prompt,
            "context": content,
            "result": result,
        }