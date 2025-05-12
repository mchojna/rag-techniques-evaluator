import os
import tempfile
from typing import List, Dict
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import SecretStr

from models.adaptive_rag import AdaptiveRAG
from models.basic_rag import BasicRAG
from models.context_window_enhancement_rag import ContextWindowEnhancementRag
from models.fusion_rag import FusionRAG
from models.graph_rag import GraphRAG
from models.semantic_chunking_rag import SemanticChunkingRAG
from tools import load_yaml_config, get_rag_techniques, get_prompt_techniques

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


def rewrite_prompt(template, examples, prompt_technique, prompt):
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "examples": RunnablePassthrough(),
            "prompt_technique": RunnablePassthrough(),
            "prompt": RunnablePassthrough(),
        }
        | prompt_template
        | ChatOpenAI(model="gpt-4o-mini")
    )
    result = chain.invoke(
        {
            "examples": "\t".join(examples),
            "prompt_technique": prompt_technique,
            "prompt": prompt,
        }
    ).content

    return result


def create_prompt(user_question, prompt_technique):
    prompts_config = load_yaml_config("config/prompts.yaml")
    prompt_techniques = get_prompt_techniques(prompts_config)

    template = """
        **Role:** You are an expert Prompt Engineer specializing in refining and optimizing prompts based on established techniques and stylistic examples.
        
        **Objective:** Rewrite the provided `{{prompt}}` according to the specified `{{prompt_technique}}`. The rewritten prompt must accurately reflect the style, structure, tone, and underlying intent demonstrated in the provided `{{examples}}`.
        
        **Context:**
        * The original `{{prompt}}` needs improvement or adaptation.
        * The `{{examples}}` serve as a gold standard for the desired output format and style.
        * The `{{prompt_technique}}` provides the specific method or framework to use for the rewrite (e.g., Chain-of-Thought, Role Prompting, Few-Shot, Zero-Shot CoT, etc.).
        
        **Instructions:**
        1.  **Analyze:** Carefully examine the `{{prompt}}`, `{{examples}}`, and `{{prompt_technique}}`.
        2.  **Identify Core Intent:** Understand the fundamental goal of the original `{{prompt}}`.
        3.  **Emulate Style:** Discern the key characteristics (formatting, language, level of detail) of the `{{examples}}`.
        4.  **Apply Technique:** Integrate the principles of the specified `{{prompt_technique}}` into the prompt structure.
        5.  **Rewrite:** Construct the new prompt, ensuring it:
            * Achieves the original intent.
            * Conforms to the style and structure of the `{{examples}}`.
            * Effectively implements the `{{prompt_technique}}`.
            * Maintains clarity and purpose.
        
        **Input Variables:**
        * `prompt`: The original prompt text to be rewritten.
        * `examples`: One or more examples demonstrating the target style and structure.
        * `prompt_technique`: The name or description of the prompt engineering technique to apply.
        
        **Output Requirements:**
        * Return **only** the final, rewritten prompt text.
        * Do not include any explanations, unnecessary content, introductory phrases, apologies, or markdown formatting around the final prompt itself (unless the examples inherently require such formatting).
        
        --- START OF INPUTS ---
        Prompt: {{{prompt}}}
        Examples: {{{examples}}}
        Prompt Technique: {{{prompt_technique}}}
        --- END OF INPUTS ---
        
        **Rewritten Prompt:**
        ```
    """

    if prompt_technique == "a-b-testing-prompt":
        # TODO
        result = user_question
    elif prompt_technique == "iterative-prompt":
        # TODO
        result = user_question
    elif prompt_technique == "ambiguity-clarity-prompt":
        # TODO
        result = user_question
    elif prompt_technique == "self-consistency-prompt":
        # TODO
        result = user_question
    elif prompt_technique in ["task-decomposition-prompt"]:
        # TODO
        result = user_question
    else:
        examples = prompts_config["prompts"][prompt_technique]
        return rewrite_prompt(template, examples, prompt_technique, user_question)

    return result


def create_rag(
    rag_technique,
    base_model,
    embedding_model,
    data,
    retriever_k,
    chunk_size,
    chunk_overlap,
):
    if rag_technique == "adaptive-retrieval":
        return AdaptiveRAG(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif rag_technique == "context-enrichment-window-around-chunk":
        return ContextWindowEnhancementRag(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif rag_technique == "fusion-retrieval":
        return FusionRAG(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif rag_technique == "graph-rag":
        return GraphRAG(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif rag_technique == "semantic-chunking":
        return SemanticChunkingRAG(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif rag_technique == "simple-rag":
        return BasicRAG(
            base_model=base_model,
            embedding_model=embedding_model,
            open_ai_key=OPENAI_API_KEY,
            paths=data,
            retriever_k=retriever_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    # TODO
    # if rag_technique == "hyde-hypothetical-document-embedding":
    #     pass
    # elif rag_technique == "hype-hypothetical-prompt-embeddings":
    #     pass
    # elif rag_technique == "microsoft-graphrag":
    #     pass
    # elif rag_technique == "choose-chunk-size":
    #     pass
    # elif rag_technique == "contextual-chunk-headers":
    #     pass
    # elif rag_technique == "contextual-compression":
    #     pass
    # elif rag_technique == "crag":
    #     pass
    # elif rag_technique == "dartboard":
    #     pass
    # elif rag_technique == "document-augmentation":
    #     pass
    # elif rag_technique == "explainable-retrieval":
    #     pass
    # elif rag_technique == "hierarchical-indices":
    #     pass
    # elif rag_technique == "multi-model-rag-with-captioning":
    #     pass
    # elif rag_technique == "multi-model-rag-with-colpal":
    #     pass
    # elif rag_technique == "proposition-chunking":
    #     pass
    # elif rag_technique == "query-transformations":
    #     pass
    # elif rag_technique == "raptor":
    #     pass
    # elif rag_technique == "relevant-segment-extraction":
    #     pass
    # elif rag_technique == "reliable-rag":
    #     pass
    # elif rag_technique == "reranking":
    #     pass
    # elif rag_technique == "retrieval-with-feedback-loop":
    #     pass
    # elif rag_technique == "self-rag":
    #     pass
    # elif rag_technique == "simple-csv-rag":
    #     pass

    return None


def create_files(uploaded_files):
    data_dir = "data"
    saved_paths = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            saved_paths.append(file_path)

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            except Exception as e:
                pass

    return saved_paths


def evaluate_model(
    user_question: str,
    ground_truth: str,
    evaluation_metrics: List[bool],
    visualization: bool,
    model: Dict,
    knowledge_source: List[str],
):
    prompt = create_prompt(user_question, model["prompt_technique"])
    data = create_files(knowledge_source)
    print(data)
    rag = create_rag(
        rag_technique=model["rag_technique"],
        base_model=model["model_choice"],
        embedding_model=EMBEDDING_MODEL,
        data=data,
        retriever_k=model["retriever_k"],
        chunk_size=model["chunk_size"],
        chunk_overlap=model["chunk_overlap"],
    )

    print(rag(prompt))


if __name__ == "__main__":
    pass
