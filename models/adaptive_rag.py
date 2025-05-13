from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.rag import RAG

class AdaptiveRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str, paths: str, retriever_k: int = 2, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.paths = paths
        self.retriever_k = retriever_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vectorstore = self.create_vectorstore()
        self.classify_chain = self.create_classify_chain()

        self.strategies = {
            "Factual": self.retrieve_factual,
            "Analytical": self.retrieve_analytical,
            "Opinion": self.retrieve_opinion,
            "Contextual": self.retrieve_contextual,
        }

        self.prompt = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
            {context}
    
            Question: {question}
            Answer:
        """

        self.prompt_template = PromptTemplate(
            template=self.prompt,
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

        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def create_classify_chain(self):
        llm = ChatOpenAI(
            model = self.base_model,
            temperature=0.9,
            api_key = self.open_ai_key,
        )

        prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}\nCategory:"
        )

        classify_chain = prompt | llm.with_structured_output(CategoriesOptions)
        return classify_chain

    def retrieve_factual(self, prompt: str) -> List[Document]:
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )

        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke({"query": prompt}).content

        docs = self.vectorstore.similarity_search(enhanced_query, k=self.retriever_k)

        ranking_prompt = PromptTemplate(
            input_variables=["query", 'docs'],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:self.retriever_k]]

    def retrieve_analytical(self, prompt: str) -> List[Document]:
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )
        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)

        input_data = {"query": prompt, "k": self.retriever_k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries

        docs = []
        for sub_query in sub_queries:
            docs.extend(self.vectorstore.similarity_search(sub_query, k=self.retriever_k))

        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\n
                Return only the indices of selected documents as a list of integers.
            """
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(docs)])
        input_data = {"query": prompt, "docs": docs_text, "k": self.retriever_k}
        selected_indices_results = diversity_chain.invoke(input_data).indices

        return[docs[i] for i in selected_indices_results if i < len(docs)]

    def retrieve_opinion(self, prompt: str) -> List[Document]:

        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": prompt, "k": self.retriever_k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split("\n")

        docs = []
        for viewpoint in viewpoints:
            docs.extend(self.vectorstore.similarity_search(f"{prompt}, {viewpoint}", k=self.retriever_k))

        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\nDocuments: {docs}\nSelected indices:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(docs)])
        input_data = {"query": prompt, "docs": docs_text, "k": self.retriever_k}
        selected_indices_results = opinion_chain.invoke(input_data).indices

        return [docs[int(i)] for i in selected_indices_results]

    def retrieve_contextual(self, prompt: str, user_context: str = None) -> List[Document]:

        contextual_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}"
        )

        context_chain = contextual_prompt | self.llm
        input_data = {"query": prompt, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content

        docs = self.vectorstore.similarity_search(contextualized_query, k=self.retriever_k*2)

        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10:\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        for doc in docs:
            input_data = {"query": contextualized_query, "context": user_context or "No specific context provided", "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:self.retriever_k]]

    def create_chain(self):
        chain = self.prompt_template | self.llm
        return chain

    def __call__(self, prompt: str) -> Dict:
        category = self.classify_chain.invoke({"query": prompt}).category
        strategy = self.strategies[category]
        context = "\n\n".join([doc.page_content for doc in strategy(prompt)])

        result = self.chain.invoke({"question": prompt, "context": context})

        return {
            "query": prompt,
            "context": context,
            "result": result,
        }

class CategoriesOptions(BaseModel):
    category: str = Field(
        description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual",
        # examples=["Factual", "Analytical"]
    )

class RelevantScore(BaseModel):
    score: float = Field(
        description="The relevance score of the document to the query.",
        # examples=[8.0, 0.4]
    )

class SelectedIndices(BaseModel):
    indices: List[int] = Field(
        description="The indices of the selected documents.",
        # examples=[0, 1, 2, 3]
    )

class SubQueries(BaseModel):
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis.",
        # examples=["What is the population of New York?", "What is the GDP of New York?"]
    )
