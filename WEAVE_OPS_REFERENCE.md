# Weave Ops Quick Reference

## All Traced Operations by File

### 📄 src/llamaindex/extractor.py
```python
@weave.op()
def get_extraction_agent(agent_name, data_schema)
    # Creates or retrieves LlamaExtract agent

@weave.op()
def _prepare_source(source)
    # Prepares file/stream for extraction

@weave.op()
async def extract_documents(sources, agent_name)
    # Main extraction pipeline
```

---

### 📦 src/rag/chunker.py
```python
@weave.op()
def chunk_text_by_line(text)
    # Splits text by newlines

@weave.op()
def chunk_text_with_overlap(text, chunk_size, overlap)
    # Creates overlapping chunks

@weave.op()
def chunk_text_by_paragraph(text, min_length)
    # Chunks by paragraph boundaries
```

---

### 🔢 src/rag/embed.py
```python
@weave.op()
def create_embeddings(texts, model)
    # Creates OpenAI embeddings
```

---

### 🗄️ src/rag/vectore_store.py
```python
class PineconeVectorStore:
    @weave.op()
    def initialize_index(dimension, metric, cloud, region)
        # Creates/connects to Pinecone index
    
    @weave.op()
    def upsert(vectors, namespace)
        # Uploads vectors to Pinecone
    
    @weave.op()
    def query(vector, top_k, namespace, include_metadata)
        # Queries similar vectors
    
    @weave.op()
    def describe_index_stats()
        # Gets index statistics
```

---

### 🔍 src/rag/retriever.py
```python
class Retriever:
    @weave.op()
    def preprocess_query(query)
        # Cleans and normalizes query
    
    @weave.op()
    def retrieve(query, top_k, namespace, preprocess)
        # Main retrieval pipeline
    
    @weave.op()
    def retrieve_and_rerank(query, top_k, rerank_top_n, namespace)
        # Retrieval with reranking
```

---

### 🤖 src/weave/model.py
```python
class RagModel(weave.Model):
    @weave.op()
    async def load_and_process_document(source_input, document_type)
        # Full document processing pipeline
    
    @weave.op()
    def chunk_document(text)
        # Chunks text using model config
    
    @weave.op()
    def embed_and_load(chunks, source_id)
        # Creates embeddings and uploads to Pinecone
    
    @weave.op()
    def predict(query, top_k)
        # Main RAG pipeline: retrieve + generate
    
    @weave.op()
    def generate_answer(query, context)
        # LLM answer generation
    
    @weave.op()
    def get_index_stats()
        # Returns Pinecone statistics
```

---

### 📊 src/evaluation/dataset_creator.py
```python
@weave.op()
def generate_synthetic_financial_documents()
    # Creates synthetic financial documents

@weave.op()
def create_qa_pairs_from_documents(documents)
    # Generates question-answer pairs

@weave.op()
def create_synthetic_evaluation_dataset(index_name, namespace)
    # Full dataset creation and upload pipeline
```

---

### 🧪 src/evaluation/evaluator.py
```python
@weave.op()
def create_evaluation_dataset(index_name, namespace)
    # Creates and returns Weave dataset
```

---

### 📈 src/evaluation/weave_native_eval.py
```python
# Metric Functions
@weave.op()
def recall_at_k(retrieved_docs, relevant_docs, k)
    # Calculates recall metric

@weave.op()
def mrr_at_k(retrieved_docs, relevant_docs, k)
    # Mean Reciprocal Rank

@weave.op()
def numeric_consistency(generated_answer, context)
    # Checks numeric accuracy

@weave.op()
def simple_faithfulness_scorer(output)
    # Checks answer faithfulness

# Weave Scorer Wrappers
@weave.op()
def recall_at_k_scorer(relevant_context, output)

@weave.op()
def mrr_at_k_scorer(relevant_context, output)

@weave.op()
def numeric_consistency_scorer(output)

# Helper Functions
@weave.op()
def process_query(example)
    # Preprocesses dataset examples for evaluation
```

---

## Tracing Flows

### 📝 Document Upload Flow
```
streamlit_app.py: upload_file
    ↓
extract_documents()
    ↓
get_extraction_agent()
_prepare_source()
    ↓
[Structured data extracted]
```

### 🔄 Document Processing Flow
```
load_and_process_document()
    ↓
extract_documents()
    ↓
chunk_document()
    ├── chunk_text_with_overlap()
    ↓
embed_and_load()
    ├── create_embeddings()
    ↓
    └── upsert()
```

### 🔎 Query Flow
```
predict()
    ↓
retrieve()
    ├── preprocess_query()
    ├── create_embeddings()
    └── query()
    ↓
generate_answer()
    └── [OpenAI API call]
```

### 🧪 Evaluation Flow
```
create_evaluation_dataset()
    ↓
create_synthetic_evaluation_dataset()
    ├── generate_synthetic_financial_documents()
    ├── create_qa_pairs_from_documents()
    ├── chunk_text_with_overlap()
    ├── create_embeddings()
    └── upsert()
    ↓
run_weave_native_evaluation()
    ├── RagModel.predict() [for each example]
    ├── recall_at_k_scorer()
    ├── mrr_at_k_scorer()
    ├── numeric_consistency_scorer()
    └── simple_faithfulness_scorer()
```

---

## Common Patterns

### Pattern 1: Class Methods as Ops
```python
class MyClass:
    @weave.op()
    def my_method(self, arg):
        # Traced as: MyClass.my_method
        pass
```

### Pattern 2: Async Ops
```python
@weave.op()
async def async_function(arg):
    # Fully supported, traces async execution
    result = await some_async_call()
    return result
```

### Pattern 3: Nested Ops
```python
@weave.op()
def parent_op():
    result = child_op()  # Child appears nested in trace
    return result

@weave.op()
def child_op():
    return "data"
```

### Pattern 4: Model as Op Container
```python
class MyModel(weave.Model):
    # Config attributes (versioned)
    param1: str
    param2: int
    
    @weave.op()
    def predict(self, input):
        # Every predict call is traced
        # Config changes create new versions
        pass
```

---

## Tips for Effective Tracing

### ✅ Add @weave.op() when:
- Function represents a logical pipeline step
- You want to measure performance
- Function calls external APIs
- Function is part of model evaluation
- You need input/output visibility

### ❌ Skip @weave.op() when:
- Simple utility functions (formatting, parsing)
- Functions called in tight loops (>1000x/sec)
- Pure data transformations with no side effects
- Internal helper methods

### 🎯 Best Practices:
1. **Descriptive names**: Function names appear in traces
2. **Logical grouping**: Use classes to group related ops
3. **Return meaningful data**: Return values are logged
4. **Handle errors gracefully**: Exceptions are captured in traces
5. **Use type hints**: Helps with trace visualization

---

## Viewing Traces

### Weave Dashboard Navigation:
```
https://wandb.ai/<entity>/solution-accelerator-mrm-eval/weave

Tabs:
├── Calls       → All function calls, searchable/filterable
├── Traces      → Visualize call trees and dependencies  
├── Models      → Browse model versions and configs
├── Evaluations → Compare evaluation runs
└── Datasets    → View versioned datasets
```

### Filtering Traces:
- **By Operation**: Filter by function name (e.g., "predict")
- **By Time**: Date range picker
- **By Status**: Success vs. errors
- **By Duration**: Find slow operations
- **By Input**: Search by input values

---

## Debugging with Traces

### Example: Finding Slow Operations
1. Go to Calls tab
2. Sort by duration (descending)
3. Click on slow call → see full trace tree
4. Identify bottleneck operation
5. View inputs/outputs to understand why

### Example: Debugging Failed Retrieval
1. Filter Calls by operation: "retrieve"
2. Filter by status: "error"
3. Click failed call
4. View input query and error message
5. Check retrieved results (if partial)
6. Trace back to see what query embedding was created

### Example: Comparing Model Versions
1. Go to Models tab
2. Click on "RagModel"
3. See all versions with config diffs
4. Click "Compare" between two versions
5. View performance metrics side-by-side

---

## Advanced: Programmatic Access

### Query Traces via API
```python
import weave

# Initialize client
client = weave.init("solution-accelerator-mrm-eval")

# Get recent calls to a specific operation
calls = client.calls(
    filter={"op_name": "predict"},
    limit=100
)

for call in calls:
    print(f"Query: {call.inputs['query']}")
    print(f"Latency: {call.summary['latency_ms']}ms")
    print(f"Result: {call.output['generated_answer']}")
```

### Get Model Versions
```python
# List all versions of RagModel
versions = client.models("RagModel")

for version in versions:
    print(f"Version: {version.version}")
    print(f"Config: {version.config}")
    print(f"Created: {version.created_at}")
```

---

## Performance Monitoring

### Key Metrics to Track:
```
Operation            | Target Latency | What to Watch
---------------------|----------------|---------------------------
extract_documents    | <3s            | LlamaIndex API calls
create_embeddings    | <500ms         | OpenAI API, batch size
upsert               | <200ms         | Pinecone network latency
retrieve             | <1s            | Query complexity, top_k
generate_answer      | <2s            | Context length, model speed
predict (full)       | <3s            | End-to-end user experience
```

### Set Up Alerts:
```python
# Example: Alert if predict() takes >5s
# (Set up in W&B dashboard under Alerts)

if latency_ms > 5000:
    send_alert("Slow RAG prediction detected")
```

---

*Last Updated: 2025-10-30*
*Project: OCR-MRM Solution Accelerator*

