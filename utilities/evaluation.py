import os
from typing import List, Dict, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ContextEntityRecall, NoiseSensitivity, \
    ResponseRelevancy, Faithfulness

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL", "text-embedding-ada-002")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


async def prepare_evaluation(user_input: str, response: str, retrieved_contexts: Union[List[str], str], reference: str, evaluation_metrics: Dict) -> Dict[str, float]:
    if isinstance(retrieved_contexts, str):
        retrieved_contexts = [retrieved_contexts]

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=BASE_MODEL)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )

    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    metrics = {}

    metric_classes = {
        "context_precision": (LLMContextPrecisionWithReference, {"llm": evaluator_llm}),
        "context_recall": (LLMContextRecall, {"llm": evaluator_llm}),
        "context_entities_recall": (ContextEntityRecall, {"llm": evaluator_llm}),
        "noise_sensitivity": (NoiseSensitivity, {"llm": evaluator_llm}),
        "response_relevancy": (ResponseRelevancy, {"llm": evaluator_llm, "embeddings": evaluator_embeddings}),
        "faithfulness": (Faithfulness, {"llm": evaluator_llm}),
        # TODO "discriminator":
    }

    for name, (cls, kwargs) in metric_classes.items():
        if evaluation_metrics.get(name, False):
            instance = cls(**kwargs)
            metrics[name] = await instance.single_turn_ascore(sample)

    return metrics
