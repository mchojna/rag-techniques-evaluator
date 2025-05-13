from typing import List

from models.adaptive_rag import AdaptiveRAG
from models.rag import RAG
from models.simple_rag import SimpleRAG
from models.context_window_enhancement_rag import ContextWindowEnhancementRAG
from models.fusion_rag import FusionRAG
from models.graph_rag import GraphRAG
from models.semantic_chunking_rag import SemanticChunkingRAG


def create_rag(rag_technique: str, base_model: str, embedding_model: str, open_ai_key: str, paths: List[str]|str, retriever_k: int, chunk_size: int, chunk_overlap: int) -> RAG:
    rag_classes = {
        "adaptive-rag": AdaptiveRAG,
        "context-window-enhancement-rag": ContextWindowEnhancementRAG,
        "fusion-rag": FusionRAG,
        "graph-rag": GraphRAG,
        "semantic-chunking-rag": SemanticChunkingRAG,
        "simple-rag": SimpleRAG,
        # TODO
        # "hyde-hypothetical-document-embedding-rag": HydeRAG,
        # "hype-hypothetical-prompt-embeddings-rag": HypeRAG,
        # "microsoft-graph-rag": MicrosoftGraphRAG,
        # "choose-chunk-size-rag": ChooseChunkSizeRAG,
        # "contextual-chunk-headers-rag": ContextualChunkHeadersRAG,
        # "contextual-compression-rag": ContextualCompressionRAG,
        # "crag": CRAG,
        # "dartboard-rag": DartboardRAG,
        # "document-augmentation-rag": DocumentAugmentationRAG,
        # "explainable-retrieval-rag": ExplainableRetrievalRAG,
        # "hierarchical-indices-rag": HierarchicalIndicesRAG,
        # "multi-model-with-captioning-rag": MultiModelRAGCaptioning,
        # "multi-model-with-colpal-rag": MultiModelRAGColpal,
        # "proposition-chunking-rag": PropositionChunkingRAG,
        # "query-transformations-rag": QueryTransformationsRAG,
        # "raptor-rag": RaptorRAG,
        # "relevant-segment-extraction-rag": RelevantSegmentExtractionRAG,
        # "reliable-rag": ReliableRAG,
        # "reranking-rag": RerankingRAG,
        # "with-feedback-loop-rag": RetrievalWithFeedbackLoopRAG,
        # "self-rag": SelfRAG,
        # "simple-csv-rag": SimpleCSVRAG,
    }

    return rag_classes.get(rag_technique)(
        base_model=base_model,
        embedding_model=embedding_model,
        open_ai_key=open_ai_key,
        paths=paths,
        retriever_k=retriever_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )