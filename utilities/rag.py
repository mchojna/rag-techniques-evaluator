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
        # TODO "hyde-hypothetical-document-embedding-rag": HydeRAG,
        # TODO "hype-hypothetical-prompt-embeddings-rag": HypeRAG,
        # TODO "microsoft-graph-rag": MicrosoftGraphRAG,
        # TODO "choose-chunk-size-rag": ChooseChunkSizeRAG,
        # TODO "contextual-chunk-headers-rag": ContextualChunkHeadersRAG,
        # TODO "contextual-compression-rag": ContextualCompressionRAG,
        # TODO "crag": CRAG,
        # TODO "dartboard-rag": DartboardRAG,
        # TODO "document-augmentation-rag": DocumentAugmentationRAG,
        # TODO "explainable-retrieval-rag": ExplainableRetrievalRAG,
        # TODO "hierarchical-indices-rag": HierarchicalIndicesRAG,
        # TODO "multi-model-with-captioning-rag": MultiModelRAGCaptioning,
        # TODO "multi-model-with-colpal-rag": MultiModelRAGColpal,
        # TODO "proposition-chunking-rag": PropositionChunkingRAG,
        # TODO "query-transformations-rag": QueryTransformationsRAG,
        # TODO "raptor-rag": RaptorRAG,
        # TODO "relevant-segment-extraction-rag": RelevantSegmentExtractionRAG,
        # TODO "reliable-rag": ReliableRAG,
        # TODO "reranking-rag": RerankingRAG,
        # TODO "with-feedback-loop-rag": RetrievalWithFeedbackLoopRAG,
        # TODO "self-rag": SelfRAG,
        # TODO "simple-csv-rag": SimpleCSVRAG,
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