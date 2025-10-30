# Weave Tracing Implementation Summary

## Overview
This document summarizes the comprehensive Weave.Op tracing implementation across the entire OCR-MRM RAG pipeline. All critical functions are now decorated with `@weave.op()` to enable end-to-end tracing and observability.

## What is Weave Tracing?

Weave Ops (`@weave.op()`) are functions that are:
- **Automatically versioned**: Any change to code, config, or inputs creates a new version
- **Tracked**: Every call is logged with inputs, outputs, and performance metrics
- **Traceable**: Full call tree visualization from beginning to end
- **Analyzable**: Query and compare runs in the Weave UI

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     WEAVE TRACING PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

1. DOCUMENT INGESTION
   ├── extract_documents()              [extractor.py]
   ├── get_extraction_agent()           [extractor.py]
   └── _prepare_source()                [extractor.py]
                    ↓
2. TEXT PROCESSING
   ├── chunk_text_with_overlap()        [chunker.py]
   ├── chunk_text_by_line()             [chunker.py]
   └── chunk_text_by_paragraph()        [chunker.py]
                    ↓
3. EMBEDDING CREATION
   └── create_embeddings()              [embed.py]
                    ↓
4. VECTOR STORAGE
   ├── initialize_index()               [vectore_store.py]
   ├── upsert()                         [vectore_store.py]
   ├── query()                          [vectore_store.py]
   └── describe_index_stats()           [vectore_store.py]
                    ↓
5. RETRIEVAL
   ├── preprocess_query()               [retriever.py]
   ├── retrieve()                       [retriever.py]
   └── retrieve_and_rerank()            [retriever.py]
                    ↓
6. GENERATION (RAG Model)
   ├── load_and_process_document()      [model.py]
   ├── chunk_document()                 [model.py]
   ├── embed_and_load()                 [model.py]
   ├── predict()                        [model.py]
   ├── generate_answer()                [model.py]
   └── get_index_stats()                [model.py]
                    ↓
7. EVALUATION
   ├── generate_synthetic_financial_documents()  [dataset_creator.py]
   ├── create_qa_pairs_from_documents()          [dataset_creator.py]
   ├── create_synthetic_evaluation_dataset()     [dataset_creator.py]
   ├── create_evaluation_dataset()               [evaluator.py]
   ├── recall_at_k()                             [weave_native_eval.py]
   ├── mrr_at_k()                                [weave_native_eval.py]
   ├── numeric_consistency()                     [weave_native_eval.py]
   ├── simple_faithfulness_scorer()              [weave_native_eval.py]
   ├── recall_at_k_scorer()                      [weave_native_eval.py]
   ├── mrr_at_k_scorer()                         [weave_native_eval.py]
   ├── numeric_consistency_scorer()              [weave_native_eval.py]
   └── process_query()                           [weave_native_eval.py]
```

## Files Modified

### 1. **src/llamaindex/extractor.py**
**Added Decorators:**
- `@weave.op()` on `get_extraction_agent()` - Tracks agent creation/retrieval
- `@weave.op()` on `_prepare_source()` - Tracks source preparation (files/streams)
- `@weave.op()` on `extract_documents()` - Tracks full document extraction pipeline

**Tracing Benefit:** Complete visibility into LlamaIndex extraction process, including:
- Which documents are being processed
- Extraction agent configuration
- Structured data extraction results
- Performance metrics for OCR/extraction

---

### 2. **src/rag/vectore_store.py**
**Added Decorators:**
- `@weave.op()` on `initialize_index()` - Tracks Pinecone index creation
- `@weave.op()` on `upsert()` - Tracks vector uploads to Pinecone
- `@weave.op()` on `query()` - Tracks vector similarity searches
- `@weave.op()` on `describe_index_stats()` - Tracks index statistics calls

**Tracing Benefit:** Full observability of vector database operations:
- Number of vectors upserted
- Query performance and results
- Index configuration and statistics
- Namespace usage patterns

---

### 3. **src/rag/retriever.py**
**Added Decorators:**
- `@weave.op()` on `preprocess_query()` - Tracks query preprocessing logic
- `@weave.op()` on `retrieve()` - Tracks main retrieval pipeline
- `@weave.op()` on `retrieve_and_rerank()` - Tracks enhanced retrieval with reranking

**Tracing Benefit:** Deep insights into retrieval quality:
- Query transformations
- Retrieval scores and matches
- Reranking effectiveness
- Top-k document selection

---

### 4. **src/weave/model.py**
**Added Decorators:**
- `@weave.op()` on `generate_answer()` - Tracks LLM answer generation

**Already Had Decorators:**
- `@weave.op()` on `load_and_process_document()` - Full document processing
- `@weave.op()` on `chunk_document()` - Text chunking
- `@weave.op()` on `embed_and_load()` - Embedding and storage
- `@weave.op()` on `predict()` - Main RAG prediction pipeline
- `@weave.op()` on `get_index_stats()` - Index statistics

**Bug Fixed:**
- Fixed undefined `is_text_input` variable on line 118
- Changed to: `source_id = getattr(source_input, 'name', str(source_input))`

**Tracing Benefit:** Complete RAG pipeline observability:
- Token usage and costs
- Latency at each stage
- Context used for generation
- Model configuration versioning

---

### 5. **src/rag/chunker.py** ✅ Already Complete
**Existing Decorators:**
- `@weave.op()` on `chunk_text_by_line()`
- `@weave.op()` on `chunk_text_with_overlap()`
- `@weave.op()` on `chunk_text_by_paragraph()`

**Tracing Benefit:** Analyze chunking strategy effectiveness

---

### 6. **src/rag/embed.py** ✅ Already Complete
**Existing Decorators:**
- `@weave.op()` on `create_embeddings()`

**Tracing Benefit:** Track embedding API calls, costs, and performance

---

### 7. **src/evaluation/dataset_creator.py**
**Added Decorators:**
- `@weave.op()` on `generate_synthetic_financial_documents()` - Tracks synthetic data generation
- `@weave.op()` on `create_qa_pairs_from_documents()` - Tracks QA pair creation

**Already Had Decorators:**
- `@weave.op()` on `create_synthetic_evaluation_dataset()` - Full dataset creation

**Tracing Benefit:** Track evaluation dataset creation and versioning

---

### 8. **src/evaluation/evaluator.py**
**Added Decorators:**
- `@weave.op()` on `create_evaluation_dataset()` - Tracks evaluation dataset setup

**Tracing Benefit:** Track dataset references and configurations

---

### 9. **src/evaluation/weave_native_eval.py** ✅ Already Complete
**Existing Decorators:**
- All scorer functions already decorated
- Process functions already decorated

**Tracing Benefit:** Full evaluation metrics tracking and comparison

---

## What Gets Traced?

### 📊 Automatic Metrics
For every `@weave.op()` decorated function, Weave automatically tracks:
- **Inputs**: All function parameters and their values
- **Outputs**: Return values and structure
- **Latency**: Execution time in milliseconds
- **Timestamps**: When the operation occurred
- **Call Tree**: Parent-child relationships between operations
- **Errors**: Exceptions and stack traces

### 🔍 Custom Tracking
Additionally tracked through our implementation:
- **Token Usage**: Via OpenAI API calls in `generate_answer()`
- **Retrieval Scores**: Similarity scores from Pinecone
- **Document Metadata**: File names, chunk numbers, sources
- **Model Versions**: Automatic versioning when config changes

---

## Usage Example

### Basic Query Flow with Tracing

```python
import weave

# Initialize Weave (already done in streamlit_app.py)
weave.init("solution-accelerator-mrm-eval")

# Create model (automatically versioned by Weave)
model = RagModel(
    index_name="ocr-mrm-db",
    namespace="default"
)

# Every operation is now traced!
result = model.predict(query="What is the company's revenue?")

# View traces at:
# https://wandb.ai/your-team/solution-accelerator-mrm-eval/weave
```

### What You'll See in Weave UI:

```
predict()                                    [2.3s]
├── retrieve()                              [0.8s]
│   ├── preprocess_query()                  [0.01s]
│   ├── create_embeddings()                 [0.4s]
│   └── query()                             [0.39s]
└── generate_answer()                        [1.5s]
    └── OpenAI API call                     [1.4s]
```

---

## Key Benefits

### 1. **Complete Observability**
- See every step of your RAG pipeline
- Identify bottlenecks and slow operations
- Track API costs and token usage

### 2. **Automatic Versioning**
- Every code change creates a new version
- Compare performance across versions
- Roll back to previous configurations

### 3. **Debugging & Error Tracking**
- Full stack traces for errors
- Input/output inspection for failed calls
- Identify edge cases and failure patterns

### 4. **Evaluation & Comparison**
- Compare different model configurations
- A/B test prompts and retrieval strategies
- Measure impact of chunking strategies

### 5. **MRM Compliance**
- Complete audit trail of all operations
- Track which model version produced which output
- Reproducible results with version pinning

---

## Streamlit Integration

The main application (`streamlit_app.py`) initializes Weave on startup:

```python
@st.cache_resource
def init_model():
    weave.init("solution-accelerator-mrm-eval")
    model = RagModel(
        index_name="ocr-mrm-db",
        namespace="default"
    )
    return model
```

Every user interaction is automatically traced:
- Document uploads → `extract_documents()`
- Chunking and embedding → `chunk_text_with_overlap()`, `create_embeddings()`
- Vector storage → `upsert()`
- Queries → `predict()` → `retrieve()` → `generate_answer()`

---

## Viewing Traces

### In the Weave Dashboard:

1. **Calls Tab**: See all function calls in chronological order
2. **Traces Tab**: Visualize the full call tree for each request
3. **Models Tab**: View all versions of your RagModel
4. **Evaluations Tab**: Compare evaluation runs side-by-side
5. **Datasets Tab**: Version and track evaluation datasets

### Direct Link:
```
https://wandb.ai/wandb-smle/solution-accelerator-mrm-eval/weave
```

---

## Best Practices

### ✅ DO:
- Keep `@weave.op()` on all functions that represent distinct logical steps
- Use descriptive function names (they become operation names in Weave)
- Let Weave auto-track inputs/outputs rather than manual logging
- Use `weave.Model` for configurations you want to version

### ❌ DON'T:
- Don't add decorators to simple utility functions (like formatters)
- Don't add decorators to functions called in tight loops (unless needed)
- Don't manually log what Weave already tracks
- Don't remove decorators without understanding impact on tracing

---

## Performance Impact

Weave tracing adds minimal overhead:
- **Latency**: ~5-10ms per operation
- **Storage**: Metadata only, no data stored unless you call `weave.publish()`
- **Network**: Asynchronous calls, non-blocking

For production systems, you can:
- Reduce sampling rate
- Disable tracing for specific operations
- Use conditional tracing based on environment

---

## Next Steps

### 1. **Explore Your Traces**
Run the Streamlit app and process a document, then view traces in Weave UI.

### 2. **Run Evaluations**
The evaluation pipeline is fully traced. Run an evaluation to see:
- Which documents are retrieved for each query
- How scores change across model versions
- Token costs and latency for each example

### 3. **Compare Model Versions**
Make a change to your RAG configuration (e.g., change `chunk_size` from 500 to 300) and compare performance in the Weave UI.

### 4. **Set Up Monitoring**
Use Weave's API to create alerts for:
- High latency operations
- Increased error rates
- Token usage spikes

---

## Summary

✅ **Total Operations Traced: 31+**

**By Module:**
- Extraction: 3 operations
- Chunking: 3 operations
- Embedding: 1 operation
- Vector Store: 4 operations
- Retrieval: 3 operations
- RAG Model: 6 operations
- Evaluation: 11+ operations

**Result:**
🎯 **Complete end-to-end tracing from document upload to answer generation!**

Every step in your RAG pipeline is now observable, debuggable, and version-controlled through Weave.

---

## Questions?

- **Weave Docs**: https://weave-docs.wandb.ai/
- **W&B Support**: support@wandb.ai
- **Example Projects**: https://wandb.ai/wandb/weave-examples

---

*Generated: 2025-10-30*
*Project: OCR-MRM RAG Solution Accelerator*
*Weave Project: solution-accelerator-mrm-eval*

