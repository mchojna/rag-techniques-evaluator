# RAG Techniques Evaluator

A benchmarking tool to evaluate and compare different Retrieval-Augmented Generation (RAG) techniques using standardized datasets and metrics. Designed to support research and optimization in RAG pipelines.

## Features

- **RAG Techniques**: Supports multiple RAG techniques such as Simple RAG, Adaptive RAG, Fusion RAG, Graph RAG, Semantic Chunking RAG, and Context Window Enhancement RAG.
- **Prompt Engineering**: Includes various prompt engineering techniques like zero-shot, few-shot, chain-of-thought, and more.
- **Evaluation Metrics**: Provides metrics such as context precision, context recall, response relevancy, faithfulness, and noise sensitivity.
- **Interactive Frontend**: Streamlit-based UI for configuring and running evaluations.
- **Extensibility**: Modular design to easily add new RAG techniques, prompt techniques, and evaluation metrics.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/rag-techniques-evaluator.git
   cd rag-techniques-evaluator
   ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables: Create a `.env` file in the root directory:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    BASE_MODEL=gpt-4o-mini
    EMBEDDING_MODEL=text-embedding-ada-002 
    ```

## Usage

### Running the Application

1. Start the Streamlit frontend

    ```bash
    streamlit run [frontend.py](http://_vscodecontentref_/1)
    ```

2. Open the application in your browser at <http://localhost:8501>.

3. Configure the evaluation parameters, upload knowledge sources, and run the evaluation.

### Backend API

The backend provides functions to evaluate models and generate metrics. You can use the `evaluate_model` function in `backend.py` for programmatic access.

### Adding New RAG Techniques

To add a new RAG technique:

1. Create a new Python file in the `models` directory.
2. Implement a class that inherits from RAG (defined in `models/rag.py`).
3. Register the new technique in `utilities/rag.py`.

### Adding New Prompt Techniques

To add a new prompt technique:

1. Update the `config/prompts.yaml` file with the new technique and examples.
2. Implement the logic in `utilities/prompt.py` if needed.

## Project Structure

    ```bash
    rag-techniques-evaluator/
    ├── config/                 # Configuration files for RAG and prompt techniques
    ├── models/                 # Implementation of various RAG techniques
    ├── utilities/              # Helper functions for prompts, RAG creation, and evaluation
    ├── [frontend.py](http://_vscodecontentref_/2)             # Streamlit-based frontend
    ├── [backend.py](http://_vscodecontentref_/3)              # Backend logic for evaluation
    ├── [requirements.txt](http://_vscodecontentref_/4)        # Python dependencies
    ├── [README.md](http://_vscodecontentref_/5)               # Project documentation
    ```

## Supported RAG Techniques

- Simple RAG: Basic retrieval and generation pipeline.
- Adaptive RAG: Dynamically adapts retrieval strategies based on query type.
- Fusion RAG: Combines vector-based and BM25 retrieval for better results.
- Graph RAG: Builds a knowledge graph for enhanced retrieval.
- Semantic Chunking RAG: Uses semantic chunking for document splitting.
- Context Window Enhancement RAG: Expands context windows for better retrieval.

## Supported Prompt Techniques

- Basic Prompt
- Zero-Shot Prompt
- Few-Shot Prompt
- Chain-of-Thought Prompt
- Constrained Prompt
- Rule-Based Prompt
- Role-Based Prompt
- Negative Prompt
- Exclusion Prompt

## TODO

### Features

- Implement additional RAG techniques listed in config/rags.yaml (e.g., hyde-hypothetical-document-embedding-rag, query-transformations-rag).
- Add support for self-consistency and task-decomposition prompt techniques.
- Integrate a discriminator-based evaluation metric.

### Enhancements

- Improve visualization in the frontend with more detailed charts and graphs.
- Add support for uploading non-PDF knowledge sources (e.g., text files, CSVs).
- Optimize the performance of large document processing.

### Documentation

- Add detailed examples for using each RAG technique.
- Provide a guide for extending the evaluation metrics.

### Testing

- Write unit tests for all utility functions.
- Add integration tests for the backend and frontend.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.
