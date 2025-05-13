from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
import spacy
import heapq
import numpy as np
from spacy.cli import download
from tqdm import tqdm

from models.rag import RAG

class DocumentProcessor:
    def __init__(self, embedding_model: str, open_ai_key:str, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=open_ai_key,
        )

    def process_document(self, documents):
        splits = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )
        return splits, vectorstore

    def create_embeddings_batch(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.append(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        return cosine_similarity(embeddings)

class Concepts(BaseModel):
    concepts_list: List[str] = Field(
        description="List of concepts"
    )

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        self._add_notes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_notes(self, splits):
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy mode...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self,  content, llm):
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits), desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]["concepts"] = concepts

    def _add_edges(self, embeddings):
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in tqdm(range(node1 + 1, num_nodes)):
                similarity_score = similarity_matrix[node1, node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]["concepts"]) & set(self.graph.nodes[node2]["concepts"])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts),)
    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        max_possible_shared = min(len(self.graph.nodes[node1]["concepts"]), len(self.graph.nodes[node2]["concepts"]))
        normalized_shared_score = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * normalized_shared_score + beta * similarity_score

    def lemmatize_concept(self, concept):
        return " ".join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])

class AnswerCheck(BaseModel):
    is_complete: bool = Field(
        description="Whether the current context provides a complete answer to the query",
    )
    answer: str = Field(
        description="The current answer based on the context, if any",
    )

class QueryEngine:
    def __init__(self, vectorstore, knowledge_graph, llm):
        self.vectorstore = vectorstore
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_content_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        answer_check_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        expanded_context = []
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}

        for doc in relevant_docs:
            closest_nodes = self.vectorstore.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            closest_node = next(n for n in self.knowledge_graph.graph.nodes if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0

        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]["content"]
                node_concepts = self.knowledge_graph.graph.nodes[current_node]["concepts"]

                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                print(f"Step {step} - Node {node_content}:")
                print(f"Content: {node_content[:100]}...")
                print(f"Concepts: {'. '.join(node_concepts)}")
                print("-"*50)

                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                node_concepts_set = set(self.knowledge_graph.lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data["weight"]

                        distance = current_priority / (1 / edge_weight)

                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]["content"]
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]["concepts"]

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                print(f"Step {step} - Node {neighbor_content}: (neighbor of {current_node}):")
                                print(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {'. '.join(neighbor_concepts)}")
                                print("-"*50)

                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                neighbor_concepts_set = set(self.knowledge_graph.lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                if final_answer:
                    break

        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:",
            )

            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> tuple[PromptValue | str, list[int], dict[int, str]]:
        with get_openai_callback() as callback:
            print(f"\nProcessing query: {query}")
            relevant_docs = self._retrieve_relevant_documents(query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

            if not final_answer:
                print("\nGenerating final answer...")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
                )
                response_chain = response_prompt | self.llm
                input_data = {"query": query, "context": expanded_context}
                response = response_prompt.invoke(input_data)
                final_answer = response
            else:
                print("\nComplete answer found during traversal...")

            print(f"\nFinal answer: {final_answer}")
            print(f"\nTotal tokens: {callback.total_tokens}")
            print(f"\nPrompt tokens: {callback.prompt_tokens}")
            print(f"\nCompletion tokens: {callback.completion_tokens}")
            print(f"\nTotal cost (USD): {callback.total_cost}")

        return final_answer, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str):
        print("\nRetrieving relevant documents...")
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

        return compression_retriever.invoke(query)

class GraphRAG(RAG):
    def __init__(self, base_model: str, embedding_model: str, open_ai_key: str, paths: str, retriever_k: int = 2, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(base_model, embedding_model, open_ai_key)

        self.paths = paths
        self.retriever_k = retriever_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.document_processor = DocumentProcessor(self.embedding_model, self.open_ai_key)
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None

    def process_documents(self, documents):
        splits, vectorstore = self.document_processor.process_document(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embeddings)
        self.query_engine = QueryEngine(vectorstore, self.knowledge_graph, self.llm)

    def query(self, query: str):
        response, traversal_path, filtered_content = self.query_engine.query(query)

        return response, filtered_content

    def __call__(self, prompt: str) -> Dict:
        all_documents = []
        for path in self.paths:
            loader = PyPDFLoader(path)
            documents = loader.load()
            all_documents.extend(documents)

        self.process_documents(all_documents)
        result, content = self.query(prompt)

        return {
            "query": prompt,
            "context": content,
            "result": result,
        }