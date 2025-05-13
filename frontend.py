import asyncio
import os
from dotenv import load_dotenv
import streamlit as st
import altair as alt
import pandas as pd
from backend import evaluate_model
from utilities.tools import load_yaml_config, get_rag_techniques, get_prompt_techniques

load_dotenv()


def reset_inputs():
    st.session_state.clear()
    st.rerun()


def load_config(config):
    for key, default in config.items():
        if key not in st.session_state:
            st.session_state[key] = default


def run():
    st.set_page_config(page_title="RAG Evaluator", layout="wide", page_icon="📊")

    prompt_techniques = get_prompt_techniques(load_yaml_config("config/prompts.yaml"))
    rag_techniques = get_rag_techniques(load_yaml_config("config/rags.yaml"))

    config = {
        "user_question": "",
        "ground_truth": "",
        "context_precision": False,
        "context_recall": False,
        "context_entities_recall": False,
        "noise_sensitivity": False,
        "response_relevancy": False,
        "faithfulness": False,
        "discriminator": False,
        "visualization": False,
        "model_choice_1": os.getenv("BASE_MODEL", "gpt-4o-mini"),
        "temperature_1": 0.7,
        "max_tokens_1": 500,
        "prompt_technique_1": prompt_techniques[0],
        "rag_technique_1": rag_techniques[0],
        "chunk_size_1": 500,
        "chunk_overlap_1": 200,
        "retriever_k_1": 5,
        "question_1": "Here will be your question...",
        "context_1": "Here will be the context...",
        "answer_1": "Here will be the answer...",
        "model_choice_2": os.getenv("BASE_MODEL", "gpt-4o-mini"),
        "temperature_2": 0.7,
        "max_tokens_2": 500,
        "prompt_technique_2": prompt_techniques[0],
        "rag_technique_2": rag_techniques[0],
        "chunk_size_2": 500,
        "chunk_overlap_2": 200,
        "retriever_k_2": 5,
        "question_2": "Here will be your question...",
        "context_2": "Here will be the context...",
        "answer_2": "Here will be the answer...",
    }

    load_config(config)

    st.title("RAG Techniques Evaluator")

    with st.sidebar:
        st.title("🛠️ Evaluation Setup")
        st.caption(
            "Configure the question, reference answer, supporting documents, and evaluation criteria to assess the performance of a RAG system."
        )
        st.divider()

        st.header("🧠 Evaluation Inputs")
        st.caption(
            "Provide the question you'd ask the system and the correct (reference) answer it should generate."
        )
        user_question = st.text_area(
            label="User Question",
            key="user_question",
        )
        ground_truth = st.text_area(
            label="Ground Truth Answer",
            key="ground_truth",
        )
        st.divider()

        st.header("📄 Document Upload")
        st.caption(
            "Upload supporting documents that the RAG model will use as context for generating answers."
        )
        knowledge_source = st.file_uploader(
            label="Knowledge Source",
            accept_multiple_files=True,
            type="pdf",
            key="knowledge_source",
        )

        st.header("📊 Evaluation Metrics")
        st.caption(
            "Select the metrics you'd like to use for evaluating the system's response against the reference answer and context."
        )
        context_precision = st.checkbox(
            label="Context Precision",
            help="How precisely the retrieved context supports the correct answer.",
            key="context_precision",
        )
        context_recall = st.checkbox(
            label="Context Recall",
            help="How completely the relevant parts of the context are retrieved.",
            key="context_recall",
        )
        context_entities_recall = st.checkbox(
            label="Context Entities Recall",
            help="Measures if key entities from the reference are present in the context.",
            key="context_entities_recall",
        )
        noise_sensitivity = st.checkbox(
            label="Noise Sensitivity",
            help="Evaluates if the model is influenced by irrelevant or misleading information.",
            key="noise_sensitivity",
        )
        response_relevancy = st.checkbox(
            label="Response Relevancy",
            help="Checks how relevant the generated answer is to the question.",
            key="response_relevancy",
        )
        faithfulness = st.checkbox(
            label="Faithfulness",
            help="Assesses whether the generated answer is consistent with the provided context.",
            key="faithfulness",
        )
        discriminator = st.checkbox(
            label="Discriminator (TBD...)",
            help="Sends the generated answer to a separate LLM, which acts as a judge to determine its quality or correctness.",
            key="discriminator",
            disabled=True,
        )
        st.divider()

        st.header("📈 Visualizations")
        visualization = st.checkbox(
            label="Visualization (TBD...)",
            help="Displays a visual summary of the generated content, such as charts, graphs, or other visual aids that help interpret the information more easily.",
            key="visualization",
            disabled=True,
        )
        st.divider()

        st.header("🕹️ Control Panel")
        st.caption(
            "Run the evaluation with the selected parameters or reset all inputs to start over."
        )
        col_start, col_reset = st.columns([1, 1])
        with col_start:
            start_button = st.button("Start Evaluation", use_container_width=True)
        with col_reset:
            reset_button = st.button("Reset Inputs", use_container_width=True)

    with st.container():
        col_model_1, col_model_2 = st.columns(2)

        with col_model_1:
            st.header("Large Language Model 1")
            st.divider()

            st.subheader("🦾 LLM Settings")
            llm_model_1 = st.selectbox(
                "Select LLM Model",
                options=["gpt-4o-mini", "gpt-3.5-turbo"],
                help="Choose the LLM model you want to use.",
                key="model_choice_1",
            )
            temperature_1 = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                help="Controls the creativity of the model's responses.",
                key="temperature_1",
            )
            max_tokens_1 = st.number_input(
                "Max Tokens",
                min_value=50,
                max_value=1000,
                help="Maximum number of tokens in the response.",
                key="max_tokens_1",
            )
            st.divider()

            st.subheader("💡 RAG Settings")
            prompt_technique_1 = st.selectbox(
                "Select Prompt Technique",
                options=prompt_techniques,
                help="Prompt technique for RAG.",
                key="prompt_technique_1",
            )
            rag_technique_1 = st.selectbox(
                "Select RAG Technique",
                options=rag_techniques,
                help="Retrieval technique for RAG.",
                key="rag_technique_1",
            )
            chunk_size_1 = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                step=50,
                help="Defines the size of text chunks for processing.",
                key="chunk_size_1",
            )
            chunk_overlap_1 = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                step=10,
                help="Specifies the overlap between consecutive text chunks.",
                key="chunk_overlap_1",
            )
            retriever_k_1 = st.number_input(
                "Retriever K",
                min_value=1,
                max_value=20,
                step=1,
                help="Determines the number of relevant documents to retrieve.",
                key="retriever_k_1",
            )
            st.divider()

            st.subheader("✉️ Conversation")
            question_1 = st.text_area("Question", disabled=True, key="question_1")
            context_1 = st.text_area("Context", disabled=True, key="context_1")
            answer_1 = st.text_area("Answer", disabled=True, key="answer_1")
            st.divider()

            st.subheader("🔬 Results")

        with col_model_2:
            st.header("Large Language Model 2")
            st.divider()

            st.subheader("🤖 LLM Settings")
            llm_model_2 = st.selectbox(
                "Select LLM Model",
                options=["gpt-4o-mini", "gpt-3.5-turbo"],
                help="Choose the LLM model you want to use.",
                key="model_choice_2",
            )
            temperature_2 = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                help="Controls the creativity of the model's responses.",
                key="temperature_2",
            )
            max_tokens_2 = st.number_input(
                "Max Tokens",
                min_value=50,
                max_value=1000,
                help="Maximum number of tokens in the response.",
                key="max_tokens_2",
            )
            st.divider()

            st.subheader("🔍 RAG Settings")
            prompt_technique_2 = st.selectbox(
                "Select Prompt Technique",
                options=prompt_techniques,
                help="Prompt technique for RAG.",
                key="prompt_technique_2",
            )
            rag_technique_2 = st.selectbox(
                "Select RAG Technique",
                options=rag_techniques,
                help="Retrieval technique for RAG.",
                key="rag_technique_2",
            )
            chunk_size_2 = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                step=50,
                help="Defines the size of text chunks for processing.",
                key="chunk_size_2",
            )
            chunk_overlap_2 = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                step=10,
                help="Specifies the overlap between consecutive text chunks.",
                key="chunk_overlap_2",
            )
            retriever_k_2 = st.number_input(
                "Retriever K",
                min_value=1,
                max_value=20,
                step=1,
                help="Determines the number of relevant documents to retrieve.",
                key="retriever_k_2",
            )
            st.divider()

            st.subheader("💬 Conversation")
            question_2 = st.text_area("Question", disabled=True, key="question_2")
            context_2 = st.text_area("Context", disabled=True, key="context_2")
            answer_2 = st.text_area("Answer", disabled=True, key="answer_2")
            st.divider()

            st.subheader("📈 Results")

    if reset_button:
        reset_inputs()

    if start_button:
        with st.spinner("Evaluating models..."):
            # Run evaluations
            evaluation_model_1 = asyncio.run(
                evaluate_model(
                    user_question=user_question,
                    ground_truth=ground_truth,
                    evaluation_metrics={
                        "context_precision": context_precision,
                        "context_recall": context_recall,
                        "context_entities_recall": context_entities_recall,
                        "noise_sensitivity": noise_sensitivity,
                        "response_relevancy": response_relevancy,
                        "faithfulness": faithfulness,
                        "discriminator": discriminator,
                    },
                    model={
                        "model_choice": llm_model_1,
                        "temperature": temperature_1,
                        "max_tokens": max_tokens_1,
                        "rag_technique": rag_technique_1,
                        "prompt_technique": prompt_technique_1,
                        "chunk_size": chunk_size_1,
                        "chunk_overlap": chunk_overlap_1,
                        "retriever_k": retriever_k_1,
                    },
                    knowledge_source=knowledge_source,
                )
            )

            evaluation_model_2 = asyncio.run(
                evaluate_model(
                    user_question=user_question,
                    ground_truth=ground_truth,
                    evaluation_metrics={
                        "context_precision": context_precision,
                        "context_recall": context_recall,
                        "context_entities_recall": context_entities_recall,
                        "noise_sensitivity": noise_sensitivity,
                        "response_relevancy": response_relevancy,
                        "faithfulness": faithfulness,
                        "discriminator": discriminator,
                    },
                    model={
                        "model_choice": llm_model_2,
                        "temperature": temperature_2,
                        "max_tokens": max_tokens_2,
                        "rag_technique": rag_technique_2,
                        "prompt_technique": prompt_technique_2,
                        "chunk_size": chunk_size_2,
                        "chunk_overlap": chunk_overlap_2,
                        "retriever_k": retriever_k_2,
                    },
                    knowledge_source=knowledge_source,
                )
            )

            # Update text areas for Model 1
            st.session_state.question_1 = evaluation_model_1["question"]
            st.session_state.context_1 = (
                "\n".join(evaluation_model_1["context"])
                if isinstance(evaluation_model_1["context"], list)
                else evaluation_model_1["context"]
            )
            st.session_state.answer_1 = evaluation_model_1["answer"]

            # Update text areas for Model 2
            st.session_state.question_2 = evaluation_model_2["question"]
            st.session_state.context_2 = (
                "\n".join(evaluation_model_2["context"])
                if isinstance(evaluation_model_2["context"], list)
                else evaluation_model_2["context"]
            )
            st.session_state.answer_2 = evaluation_model_2["answer"]

            # Display evaluation results
            with col_model_1:
                st.subheader("🔬 Results")
                metrics_df_1 = pd.DataFrame(
                    evaluation_model_1["metrics"].items(), columns=["Metric", "Score"]
                )
                st.table(metrics_df_1)

                if visualization:
                    st.bar_chart(metrics_df_1.set_index("Metric"))

            with col_model_2:
                st.subheader("📈 Results")
                metrics_df_2 = pd.DataFrame(
                    evaluation_model_2["metrics"].items(), columns=["Metric", "Score"]
                )
                st.table(metrics_df_2)

                if visualization:
                    st.bar_chart(metrics_df_2.set_index("Metric"))

            # Compare models
            st.subheader("📊 Models Comparison")
            comparison_df = pd.DataFrame(
                {
                    "Metric": metrics_df_1["Metric"],
                    f"{llm_model_1}": metrics_df_1["Score"],
                    f"{llm_model_2}": metrics_df_2["Score"],
                }
            )
            st.table(comparison_df)

            if visualization:
                comparison_df_melted = comparison_df.melt(
                    id_vars=["Metric"], var_name="Model", value_name="Score"
                )
                chart = (
                    alt.Chart(comparison_df_melted)
                    .mark_bar()
                    .encode(x="Metric", y="Score", color="Model", column="Model")
                    .properties(width=300)
                )
                st.altair_chart(chart)


if __name__ == "__main__":
    run()
