import os
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utilities.tools import load_yaml_config, get_prompt_techniques

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL", "text-embedding-ada-002")


def rewrite_prompt(prompt: str, prompt_technique: str, examples: List[str]) -> str:
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
    prompt_template = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "prompt": RunnablePassthrough(),
            "prompt_technique": RunnablePassthrough(),
            "examples": RunnablePassthrough(),
        }
        | prompt_template
        | ChatOpenAI(model=BASE_MODEL)
    )

    result = chain.invoke(
        {
            "prompt": prompt,
            "prompt_technique": prompt_technique,
            "examples": "\t".join(examples),
        }
    ).content

    return result

def create_prompt(query: str, prompt_technique: str) -> str:
    prompts_config = load_yaml_config("config/prompts.yaml")
    # prompt_techniques = get_prompt_techniques(prompts_config)

    if prompt_technique == "a-b-testing-prompt":
        # TODO
        result = query
    elif prompt_technique == "iterative-prompt":
        # TODO
        result = query
    elif prompt_technique == "ambiguity-clarity-prompt":
        # TODO
        result = query
    elif prompt_technique == "self-consistency-prompt":
        # TODO
        result = query
    elif prompt_technique in ["task-decomposition-prompt"]:
        # TODO
        result = query
    else:
        examples = prompts_config["prompts"][prompt_technique]
        result = rewrite_prompt(query, prompt_technique, examples)

    return result