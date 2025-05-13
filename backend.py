import os

from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.documents import Document
from utilities.evaluation import prepare_evaluation
from utilities.prompt import create_prompt
from utilities.rag import create_rag

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

async def evaluate_model(user_question: str, ground_truth: str, evaluation_metrics: Dict, visualization: bool, model: Dict, knowledge_source: List[str]) -> Dict:
    prompt = create_prompt(user_question, model["prompt_technique"])
    print("Created prompt")
    # paths = create_files(knowledge_source)
    # print("Created paths")
    paths = knowledge_source
    rag = create_rag(
        rag_technique=model["rag_technique"],
        base_model=model["model_choice"],
        embedding_model=EMBEDDING_MODEL,
        open_ai_key=OPENAI_API_KEY,
        paths=paths,
        retriever_k=model["retriever_k"],
        chunk_size=model["chunk_size"],
        chunk_overlap=model["chunk_overlap"]
    )
    print("Created rag")

    result = rag(prompt)
    print("Created result")

    question = result["query"]
    answer = result["result"].content

    context = result["context"]

    if isinstance(context, str):
        context = context
    elif isinstance(context, Document):
        context = context.page_content
    elif isinstance(context, List):
        if isinstance(context[0], str):
            context = [doc for doc in context]
        elif isinstance(context[0], Document):
            context = [doc.page_content for doc in context]
    elif isinstance(context, Dict):
        context = list(context.values())

    metrics = await prepare_evaluation(
        user_input=question,
        response=answer,
        retrieved_contexts=context,
        reference=ground_truth,
        evaluation_metrics=evaluation_metrics
    )
    print("Created metrics")

    # TODO
    # if visualization:
    #     pass
    # print("Created visualization")

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "metrics": metrics,
    }

#
# async def main():
#     test = await evaluate_model(
#         user_question="What is the point of attention mechanism?",
#         ground_truth="The attention mechanism allows a model to focus on the most relevant parts of the input when making predictions, dynamically weighting different elements based on their importance. This improves performance, especially in tasks like machine translation and text generation, by helping the model capture dependencies and context more effectively.",
#         evaluation_metrics={
#             "context_precision": True,
#             "context_recall": True,
#             "context_entities_recall": True,
#             "noise_sensitivity": True,
#             "response_relevancy": True,
#             "faithfulness": True,
#             "discriminator": False
#         },
#         visualization=False,
#         model={
#             "model_choice": "gpt-4o-mini",
#             "temperature": 1,
#             "max_tokens": 1000,
#             "rag_technique": "simple-rag",
#             "prompt_technique": "zero-shot-prompt",
#             "chunk_size": 1000,
#             "chunk_overlap": 200,
#             "retriever_k": 2,
#         },
#         knowledge_source=[
#             "/Users/mchojna/Documents/Repozytoria/rag-techniques-evaluator/data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
#         ],
#     )
#     return test
#
# if __name__ == "__main__":
    # res = asyncio.run(main())
    # print(res)

