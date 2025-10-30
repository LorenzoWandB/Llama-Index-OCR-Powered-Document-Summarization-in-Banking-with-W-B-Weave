# Financial Document OCR RAG System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for extracting and querying financial documents, with full observability and evaluation powered by **Weights & Biases Weave**.

## Overview

This project demonstrates how to build a trustworthy AI system for financial services that combines:
- **LlamaIndex** for intelligent OCR and structured data extraction
- **OpenAI** for embeddings and language generation
- **Pinecone** for vector storage and semantic search
- **Weave** for end-to-end tracing, evaluation, and Model Risk Management (MRM) compliance

Perfect for financial institutions that need to extract, query, and validate information from income statements, balance sheets, and other financial documents.

## The Pipeline

The system follows a 4-stage RAG pipeline, with **Weave observability at every step**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────┘

1. 📄 EXTRACTION (LlamaIndex + Weave)
   └─> Financial document → LlamaIndex OCR → Structured data
       └─> @weave.op() tracks extraction process

2. 🔢 EMBEDDING & STORAGE (OpenAI + Pinecone + Weave)
   └─> Text chunks → OpenAI embeddings → Pinecone vector store
       └─> @weave.op() tracks chunking and embedding

3. 🔍 RETRIEVAL (Pinecone + Weave)
   └─> Query → Embedding → Vector search → Relevant chunks
       └─> @weave.op() tracks retrieval quality

4. 🤖 GENERATION (GPT-4 + Weave)
   └─> Query + Context → LLM → Answer
       └─> @weave.op() tracks generation and faithfulness
```

### How Weave Powers Observability

**Every operation is traced** using Weave's `@weave.op()` decorator:

```python
@weave.op()
def extract_documents(sources):
    # LlamaIndex extraction logic
    # Weave automatically logs inputs, outputs, latency
    ...

@weave.op()
def create_embeddings(texts):
    # OpenAI embedding logic
    # Weave tracks token usage, model versions
    ...

@weave.op()
def generate_answer(query, context):
    # GPT-4 generation logic
    # Weave captures full prompt, response, tokens
    ...
```

The entire `RagModel` inherits from `weave.Model`, which provides:
- **Automatic versioning** when you change parameters
- **Configuration tracking** (chunk size, embedding model, LLM settings)
- **End-to-end traces** linking extraction → embedding → retrieval → generation

## Weave for MRM & Evaluation

Beyond tracing, Weave enables rigorous evaluation for Model Risk Management:

### 1. Custom Scorers
Define metrics that matter for financial accuracy:

```python
@weave.op()
def recall_at_k(retrieved_docs, relevant_docs, k):
    """How many important documents did we retrieve?"""
    ...

@weave.op()
def numeric_consistency(answer, context):
    """Do numbers in the answer appear in source documents?"""
    ...

@weave.op()
def faithfulness(answer, context):
    """Is the answer grounded in retrieved context?"""
    ...
```

### 2. Dataset Versioning
Create reusable, versioned evaluation datasets:

```python
dataset = weave.Dataset(
    name="financial_rag_eval",
    rows=[
        {"query": "What is net profit?", "expected_docs": [...], ...},
        {"query": "What are total expenses?", ...},
    ]
)
weave.publish(dataset)  # Versioned and shareable
```

### 3. Evaluation Comparison
Run evaluations across model versions and compare:
- **Quantitative metrics**: Recall, MRR, faithfulness, numeric consistency
- **Qualitative outputs**: Side-by-side answer comparison
- **Performance**: Latency and token usage

Weave automatically tracks which model version, which dataset version, and which prompt version produced each result.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **OCR & Extraction** | LlamaIndex (LlamaExtract) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Database** | Pinecone (Serverless) |
| **LLM** | OpenAI GPT-4 |
| **Observability & Eval** | Weights & Biases Weave |
| **UI** | Streamlit |

## Setup

### 1. Prerequisites

- Python 3.13+
- API Keys for:
  - OpenAI
  - Pinecone
  - LlamaIndex Cloud

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd OCR_MRM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Pinecone
PINECONE_API_KEY=your_pinecone_key_here

# LlamaIndex Cloud
LLAMA_CLOUD_API_KEY=your_llama_cloud_key_here
```

### 4. Configure Weave Project

Update the Weave project name in `streamlit_app.py` (line 227):

```python
weave.init("your-weave-project-name")  # Change this to your project
```

### 5. Run the Demo

```bash
# Using the start script
./start.sh

# Or directly
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
OCR_MRM/
├── src/
│   ├── llamaindex/         # LlamaIndex extraction logic
│   │   ├── extractor.py    # Document OCR and structured extraction
│   │   └── llama.py        # LlamaIndex configuration
│   ├── rag/                # RAG pipeline components
│   │   ├── chunker.py      # Text chunking with overlap
│   │   ├── embed.py        # OpenAI embedding creation
│   │   ├── vectore_store.py # Pinecone vector store wrapper
│   │   └── retriever.py    # Vector search and retrieval
│   ├── weave/              # Weave model and prompts
│   │   ├── model.py        # Main RagModel (weave.Model)
│   │   └── prompt.py       # Prompt templates
│   └── evaluation/         # Evaluation framework
│       ├── dataset_creator.py   # Synthetic data generation
│       ├── evaluator.py         # Custom scorers
│       └── weave_native_eval.py # Weave evaluation runner
├── data/
│   └── images/             # Sample financial documents
├── streamlit_app.py        # Interactive demo UI
├── requirements.txt        # Python dependencies
└── start.sh               # Launch script
```

## Usage

### Demo UI (Streamlit)

The Streamlit app provides a 4-step interactive demo:

1. **Upload & Extract**: Upload a financial document, extract structured data with LlamaIndex
2. **Embed & Upsert**: Create embeddings and store in Pinecone
3. **Ask & Answer**: Query the document using natural language
4. **Evaluate**: View evaluation methodology and comparison views

### Programmatic Usage

```python
import weave
from src.weave.model import RagModel

# Initialize Weave
weave.init("your-project-name")

# Create model
model = RagModel(
    index_name="financial-docs",
    namespace="default",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o",
    chunk_size=500,
    retriever_top_k=5
)

# Process a document (async)
await model.load_and_process_document("path/to/document.pdf")

# Query the document
result = model.predict("What is the company's net profit?")
print(result["generated_answer"])
```

### Running Evaluations

```python
import weave
from src.evaluation.evaluator import create_evaluation
from src.weave.model import RagModel

weave.init("your-project-name")

# Load evaluation dataset
dataset = weave.ref("your-dataset-ref").get()

# Create model
model = RagModel(index_name="financial-docs", namespace="default")

# Run evaluation
evaluation = create_evaluation(dataset)
results = await evaluation.evaluate(model)

# Results are logged to Weave for comparison
```

## Why Weave?

For financial services, **traceability and validation are critical**:

✅ **Full Lineage**: Every prediction traces back to exact model version, prompt, and data  
✅ **Evaluation Framework**: Quantify quality with custom metrics  
✅ **Version Control**: Automatic versioning of models, datasets, and prompts  
✅ **MRM Compliance**: Provides audit trail for model risk management  
✅ **Cost Tracking**: Monitor token usage and API costs  
✅ **Performance**: Track latency across the pipeline  

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

Built with ❤️ using LlamaIndex, OpenAI, Pinecone, and Weights & Biases Weave

