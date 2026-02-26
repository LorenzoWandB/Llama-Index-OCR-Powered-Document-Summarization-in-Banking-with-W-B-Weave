import streamlit as st
import os
from pathlib import Path
import tempfile
import json
from PIL import Image
import weave
import asyncio
from datetime import datetime

from src.weave.model import RagModel
from src.llamaindex.extractor import extract_documents
from src.rag.chunker import chunk_text_with_overlap
from src.rag.embed import create_embeddings

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Document Demo",
    page_icon="📄",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Step badges */
    .step-badges {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 3rem;
    }
    
    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background-color: white;
        border-radius: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .step-badge.active {
        background-color: #000000;
        color: white;
    }
    
    .step-number {
        width: 28px;
        height: 28px;
        background-color: #000000;
        color: white;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .step-badge.active .step-number {
        background-color: white;
        color: #000000;
    }
    
    /* Card styling */
    .main-card {
        background-color: white;
        border-radius: 1rem;
        padding: 3rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin: 0 auto;
        max-width: 1200px;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 0.75rem;
        padding: 4rem 2rem;
        text-align: center;
        background-color: #fafafa;
        margin-bottom: 1.5rem;
    }
    
    .preview-area {
        background-color: #f9fafb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        font-style: italic;
        color: #6b7280;
    }
    
    .preview-area.has-content {
        min-height: 150px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #000000;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1f2937;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f9fafb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation for image slide */
    @keyframes slideInLeft {
        from {
            transform: translateX(-50px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(50px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.8s ease-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.8s ease-out;
    }
    
    .extraction-card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Weave & Model ---
@st.cache_resource
def init_model():
    weave.init("solution-accelerator-mrm-eval")
    model = RagModel(
        index_name="ocr-mrm-db",
        namespace="default"
    )
    return model

model = init_model()

# --- Helper Functions ---
def display_eval_image(image_name: str, placeholder_text: str):
    """Display evaluation image if it exists, otherwise show placeholder."""
    image_paths = [
        f"data/images/evaluations/{image_name}.png",
        f"data/images/evaluations/{image_name}.jpg",
        f"data/images/evaluations/{image_name}.jpeg"
    ]
    
    image_found = False
    for img_path in image_paths:
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            image_found = True
            break
    
    if not image_found:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**📷 {placeholder_text}**")
        st.markdown(f"*Add your image to: `data/images/evaluations/{image_name}.png`*")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Session State Initialization ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'show_evaluations' not in st.session_state:
    st.session_state.show_evaluations = False

# --- Navigation Functions ---
def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def reset_demo():
    st.session_state.step = 1
    st.session_state.extracted_data = None
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.uploaded_file = None
    st.session_state.query_result = None

# --- Header ---
st.markdown('<h1 class="main-title">OCR → Embeddings → Pinecone → LLM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Weights & Biases Weave × LlamaIndex</p>', unsafe_allow_html=True)

# Step badges
step1_class = "step-badge active" if st.session_state.step == 1 else "step-badge"
step2_class = "step-badge active" if st.session_state.step == 2 else "step-badge"
step3_class = "step-badge active" if st.session_state.step == 3 else "step-badge"
step4_class = "step-badge active" if st.session_state.step == 4 else "step-badge"

st.markdown(f"""
<div class="step-badges">
    <div class="{step1_class}">
        <span class="step-number">1</span>
        <span>Upload & Extract</span>
    </div>
    <div class="{step2_class}">
        <span class="step-number">2</span>
        <span>Embed & Upsert</span>
    </div>
    <div class="{step3_class}">
        <span class="step-number">3</span>
        <span>Ask & Answer</span>
    </div>
    <div class="{step4_class}">
        <span class="step-number">4</span>
        <span>Evaluate</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# STEP 1: UPLOAD & EXTRACT
# ============================================================================
if st.session_state.step == 1:
    st.markdown('<p class="section-title">📄 1. Add a Document and Extract Text</p>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a financial document",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # If data is extracted, show side by side with animation
        if st.session_state.extracted_data:
            col1, col2 = st.columns([1, 1], gap="large")
            
            # Left: Image with slide-in animation
            with col1:
                st.markdown('<div class="slide-in-left">', unsafe_allow_html=True)
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption="Uploaded Document")
                else:
                    st.info(f"📄 **{uploaded_file.name}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Right: Extracted data with slide-in animation
            with col2:
                st.markdown('<div class="slide-in-right">', unsafe_allow_html=True)
                st.markdown('<div class="extraction-card">', unsafe_allow_html=True)
                st.markdown("### ✨ Extracted Data")
                data = st.session_state.extracted_data['data']
                for key, value in data.items():
                    field_name = key.replace('_', ' ').title()
                    st.markdown(f"**{field_name}:** {value}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            metadata = st.session_state.extracted_data.get('metadata')
            if metadata:
                def _to_dict(obj):
                    if isinstance(obj, dict):
                        return obj
                    if hasattr(obj, 'model_dump'):
                        return obj.model_dump()
                    if hasattr(obj, '__dict__'):
                        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
                    return obj

                meta_dict = _to_dict(metadata)
                field_metadata = meta_dict.get('field_metadata') if isinstance(meta_dict, dict) else None

                if field_metadata:
                    field_entries = _to_dict(field_metadata)
                    if isinstance(field_entries, dict):
                        with st.expander("🔍 View Reasoning & Citations (LlamaExtract)"):
                            for field_name, field_info in field_entries.items():
                                info = _to_dict(field_info)
                                label = field_name.replace('_', ' ').title()
                                if isinstance(info, dict):
                                    reasoning = info.get('reasoning') or info.get('reason')
                                    citation = info.get('citation') or info.get('source') or info.get('citations')
                                    st.markdown(f"**{label}**")
                                    if reasoning:
                                        st.caption(f"💡 Reasoning: {reasoning}")
                                    if citation:
                                        cite_parts = []
                                        raw = citation if isinstance(citation, list) else [citation]
                                        for c in raw:
                                            c = _to_dict(c) if not isinstance(c, (str, int, float)) else c
                                            if isinstance(c, dict):
                                                page = c.get('page')
                                                text = c.get('matching_text', '')
                                                if page is not None or text:
                                                    cite_parts.append(f"Page {page}: \"{text}\"" if page is not None else f"\"{text}\"")
                                            else:
                                                cite_parts.append(str(c))
                                        if cite_parts:
                                            st.caption(f"📄 Citation: {'; '.join(cite_parts)}")
                                    if not reasoning and not citation:
                                        st.json(info)
                                else:
                                    st.markdown(f"**{label}:** {info}")
                                st.markdown("---")
                else:
                    with st.expander("🔍 View Extraction Metadata (LlamaExtract)"):
                        st.json(_to_dict(metadata) if isinstance(_to_dict(metadata), dict) else str(metadata))
            
            # Continue button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col2:
                if st.button("Continue →", type="primary", use_container_width=True):
                    next_step()
                    st.rerun()
        
        else:
            # Before extraction: show image centered
            if uploaded_file.type.startswith('image'):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
            
            # Extract button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Extract Data", type="primary", use_container_width=True):
                    with st.spinner("Extracting structured data from document..."):
                        try:
                            uploaded_file.seek(0)
                            results = asyncio.run(extract_documents([uploaded_file]))
                            st.session_state.extracted_data = results[0]
                            st.success("✅ Extraction complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Extraction failed: {e}")
    
    else:
        # No file uploaded yet
        st.markdown('<div class="preview-area">', unsafe_allow_html=True)
        st.markdown("*(Upload a document to begin)*")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# STEP 2: EMBEDDINGS & PINECONE
# ============================================================================
elif st.session_state.step == 2:
    st.markdown('<p class="section-title">🔢 2. Create Embeddings and Send to Pinecone</p>', unsafe_allow_html=True)
    
    # Process if not already done
    if st.session_state.chunks is None:
        with st.spinner("Creating chunks and embeddings..."):
            # Convert structured data to text
            data = st.session_state.extracted_data['data']
            text_parts = []
            for key, value in data.items():
                if value is not None:
                    field_name = key.replace('_', ' ').title()
                    text_parts.append(f"{field_name}: {value}")
            extracted_text = "\n".join(text_parts)
            
            # Chunk the text
            chunks = chunk_text_with_overlap(extracted_text, chunk_size=500, overlap=100)
            st.session_state.chunks = chunks
            
            # Create embeddings
            texts_to_embed = [chunk['text'] for chunk in chunks]
            embeddings = create_embeddings(texts_to_embed, model="text-embedding-3-small")
            st.session_state.embeddings = embeddings
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(st.session_state.chunks)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Text Chunks</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(st.session_state.embeddings[0])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Vector Dimensions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(st.session_state.embeddings)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Embeddings Generated</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display chunks
    with st.expander("📦 View Text Chunks"):
        for i, chunk in enumerate(st.session_state.chunks):
            st.markdown(f"**Chunk {i+1}** ({len(chunk['text'])} chars)")
            st.text(chunk['text'])
            if i < len(st.session_state.chunks) - 1:
                st.markdown("---")
    
    # Show sample embedding
    with st.expander("🔢 View Sample Embedding Vector"):
        st.markdown("**Vector Preview** (first 20 dimensions of chunk 1):")
        st.code(str(st.session_state.embeddings[0][:20]))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upsert to Pinecone button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📤 Send to Pinecone", type="primary", use_container_width=True):
            with st.spinner("Upserting vectors to Pinecone..."):
                try:
                    # Prepare vectors
                    vectors_to_upsert = []
                    filename = st.session_state.uploaded_file.name
                    for i, chunk in enumerate(st.session_state.chunks):
                        metadata = {
                            "text": chunk['text'],
                            "filename": filename,
                            "chunk_number": i,
                        }
                        vectors_to_upsert.append({
                            "id": f"doc_{filename}_{i}",
                            "values": st.session_state.embeddings[i],
                            "metadata": metadata
                        })
                    
                    # Upsert
                    model.retriever.vector_store.upsert(vectors_to_upsert)
                    st.success(f"✅ Successfully uploaded {len(vectors_to_upsert)} vectors to Pinecone!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue →", type="primary", use_container_width=True):
            next_step()
            st.rerun()

# ============================================================================
# STEP 3: QUERY & RESPONSE
# ============================================================================
elif st.session_state.step == 3:
    st.markdown('<p class="section-title">💬 3. Ask a Question and Get an Answer</p>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Ask a question about the document:",
        placeholder="e.g., What is the company's net profit?",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_query = st.button("🔍 Run Query", type="primary", use_container_width=True)
    
    if run_query and query:
        with st.spinner("Processing query through RAG pipeline..."):
            try:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 1. Embed the query
                st.markdown("#### 1️⃣ Query Embedding")
                query_embedding = create_embeddings(query)[0]
                st.success(f"✅ Query embedded into {len(query_embedding)}-dimensional vector")
                
                # 2. Retrieve from Pinecone
                st.markdown("#### 2️⃣ Pinecone Retrieval")
                retrieved_matches = model.retriever.retrieve(query, top_k=5)
                st.success(f"✅ Retrieved {len(retrieved_matches)} most relevant chunks")
                
                with st.expander("📚 View Retrieved Chunks"):
                    for i, match in enumerate(retrieved_matches):
                        st.markdown(f"**Match {i+1}** (Score: {match['score']:.4f})")
                        st.text(match['metadata'].get('text', 'N/A'))
                        if i < len(retrieved_matches) - 1:
                            st.markdown("---")
                
                # 3. LLM Generation (Thinking)
                st.markdown("#### 3️⃣ LLM Thinking Process")
                context = [match.get('metadata', {}).get('text', '') for match in retrieved_matches]
                
                st.info("💭 Generating response with GPT-4 based on retrieved context...")
                
                with st.expander("🧠 View Context Sent to LLM"):
                    context_str = "\n---\n".join(context)
                    st.text(context_str)
                
                # 4. Generate answer
                st.markdown("#### 4️⃣ Generated Answer")
                generated_answer = model.generate_answer(query, context)
                
                st.markdown('<div class="info-box" style="background-color: #f0fdf4; border-color: #bbf7d0;">', unsafe_allow_html=True)
                st.markdown(f"**Q:** {query}")
                st.markdown(f"**A:** {generated_answer}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("✅ Query complete!")
                st.balloons()
                
                # Store result
                st.session_state.query_result = {
                    "query": query,
                    "answer": generated_answer,
                    "retrieved_matches": len(retrieved_matches)
                }
                
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                st.exception(e)
    
    # Show previous result if exists
    elif st.session_state.query_result and not query:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Previous Query Result")
        st.markdown('<div class="info-box" style="background-color: #f0fdf4; border-color: #bbf7d0;">', unsafe_allow_html=True)
        st.markdown(f"**Q:** {st.session_state.query_result['query']}")
        st.markdown(f"**A:** {st.session_state.query_result['answer']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("← Back", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button("✨ Ask Another", use_container_width=True):
            st.session_state.query_result = None
            st.rerun()
    with col3:
        if st.button("Continue →", type="primary", use_container_width=True):
            next_step()
            st.rerun()
    with col4:
        if st.button("🔄 Start Over", use_container_width=True):
            reset_demo()
            st.rerun()

# ============================================================================
# STEP 4: EVALUATION WITH WEAVE
# ============================================================================
elif st.session_state.step == 4:
    st.markdown('<p class="section-title">📊 4. Evaluate with Weave</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Iterative Improvement Through Evaluation
     
    This is where **Weave's Evaluations** come into play. The goal is to define quantitative metrics that reflect 
    the quality and compliance of the model's outputs, then measure the pipeline against a test suite of examples. 
    By doing so iteratively, teams can tune the system to meet internal and regulatory standards, giving confidence 
    to bring these applications to production.
    """)
    
    
    # Defining Evaluation Metrics
    st.markdown("### 🎯 Defining Evaluation Metrics")
    
    st.markdown("""
    Weave allows you to define **custom Scorers** (as well as out-of-the-box scorers). These can be Python classes 
    or simple functions that evaluate an (input, output) pair and return a score. We use the following scorers:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Retrieval Metrics")
        st.markdown("""
        - **Recall@k**: How many important docs did we retrieve  
          *(k = #retrieved_chunks)*
        
        - **MRR@k**: Mean Reciprocal Rank - how high does the first relevant doc appear  
          *(k = #retrieved_chunks)*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🤖 Generation Metrics")
        st.markdown("""
        - **Faithfulness**: Whether the answer is grounded in the context  
          *(using HallucinationFreeScorer)*
        
        - **Numeric Consistency**: Regex check whether numbers appeared in the context
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How to Create Evaluations
    st.markdown("### ⚙️ Creating Evaluations in 3 Steps")
    
    with st.expander("1️⃣ Create Scorers as Functions", expanded=False):
        st.code("""
@weave.op()
def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: Optional[int] = None) -> float:
    '''Calculate Recall@k - fraction of relevant documents retrieved in top-k results.'''
    if not relevant_docs:
        return 1.0  # No relevant docs to retrieve
    
    if not retrieved_docs:
        return 0.0  # No docs retrieved
    
    # Consider only top-k retrieved documents
    top_k_retrieved = retrieved_docs[:k] if k is not None else retrieved_docs
    
    # Count how many relevant docs are in the top-k retrieved
    retrieved_relevant = 0
    for doc in top_k_retrieved:
        if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs):
            retrieved_relevant += 1
    
    return retrieved_relevant / len(relevant_docs)
        """, language="python")
    
    with st.expander("2️⃣ Create and Publish a Dataset", expanded=False):
        st.markdown("""
        You can generate synthetic data or use real data. Create and publish a referenceable, versioned Dataset:
        """)
        st.code("""
# Create and publish Weave dataset
dataset = weave.Dataset(
    name="financial_rag_eval_synthetic",
    rows=dataset_rows
)
dataset_ref = weave.publish(dataset)

# Retrieve and use a published version of a dataset
dataset = weave.ref("reference_here").get()
        """, language="python")
    
    with st.expander("3️⃣ Run the Evaluation", expanded=False):
        st.code("""
# Create evaluation with all scorers
evaluation = Evaluation(
    dataset=dataset,
    scorers=[
        recall_at_k_scorer,
        mrr_at_k_scorer, 
        numeric_consistency_scorer,
        simple_faithfulness_scorer
    ],
    preprocess_model_input=process_query,
)

# Run evaluation
evaluation_results = await evaluation.evaluate(model)
        """, language="python")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Experimentation Section
    st.markdown("### 🔬 Experiment & Iterate")
    st.markdown("""
    Run evaluations multiple times as you experiment with:
    - Changing the underlying model
    - Experimenting with number of documents or chunks retrieved
    - Changing system prompts
    - And more...
    """)
    
    st.info("💡 **Weave automatically tracks and versions** everything as you experiment—Models, Datasets, Prompts, Ops, etc. "
            "This granular traceability allows developer and MRM teams to easily link evaluation results to specific components.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison View Section
    st.markdown("### 📈 Evaluation Comparison - 3 Core Views")
    st.markdown("""
    Once you have multiple evaluations logged to Weave, you can compare them to measure the impact of your experiments. 
    Weave's evaluation comparison gives a **360-degree view** for developers to understand and measure impact:
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # View 1: Comparison Overview
    st.markdown("#### 1️⃣ Comparison Overview")
    st.markdown("""
    A single view to determine the deltas across all metrics you're evaluating against. Weave shows which specific 
    version of a Model was used for each evaluation, answering: *"How does my new version compare to baseline?"*
    """)
    
    # Display image 1
    display_eval_image("eval_comparison_overview", "[Add comparison overview screenshot here]")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # View 2: Quantitative Comparison
    st.markdown("#### 2️⃣ Quantitative Comparison")
    st.markdown("""
    Drill down into each specific metric and directly measure the quantitative difference between evaluations. 
    Weave automatically adds **Latency** and **Total Tokens**, allowing teams to balance accuracy vs. performance and cost.
    """)
    
    # Display image 2
    display_eval_image("eval_quantitative", "[Add quantitative comparison screenshot here]")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # View 3: Qualitative Comparison
    st.markdown("#### 3️⃣ Qualitative Comparison")
    st.markdown("""
    Explore each specific example to qualitatively understand what was inputted, what documents were used, and what 
    each evaluation outputted. This view allows developers to:
    - Identify edge cases where all evals fail
    - Compare outputs from different model versions
    - Debug example-level issues
    """)
    
    # Display image 3
    display_eval_image("eval_qualitative", "[Add qualitative comparison screenshot here]")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Conclusion
    st.success("""
    🎯 **The Result**: This single Evaluation Comparison view—broken down into 3 core areas—allows teams to iterate 
    and improve their prototypes to reach a level of confidence to bring these apps into production. This provides 
    a way for Financial Services organizations to develop quickly while ensuring confidence and granular traceability 
    for MRM and governance teams.
    """)
    
    # Link to Weave Dashboard
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.link_button("🔗 View Full Dashboard on Weave", 
                      "https://wandb.ai/wandb-smle/solution-accelerator-mrm-eval/weave/compare-evaluations?evaluationCallIds=%5B%22019856af-f526-7889-a5f5-df2337b3f3ad%22%2C%22019856a5-f93b-7f86-aeb5-a0474ca64d6a%22%2C%22019856a5-4274-7ddb-875e-59dde82b5a31%22%5D&metrics=%7B%22recall_at_k_scorer.recall_at_5%22%3Atrue%2C%22mrr_at_k_scorer.mrr_at_5%22%3Atrue%2C%22numeric_consistency_scorer.numeric_consistency_score%22%3Atrue%2C%22numeric_consistency_scorer.numeric_details.consistency_score%22%3Atrue%2C%22numeric_consistency_scorer.numeric_details.total_numbers%22%3Atrue%2C%22numeric_consistency_scorer.numeric_details.consistent_numbers%22%3Atrue%2C%22simple_faithfulness_scorer.faithfulness_score%22%3Atrue%2C%22Latency%22%3Atrue%2C%22Total%20Tokens%22%3Atrue%7D",
                      use_container_width=True)
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", use_container_width=True):
            reset_demo()
            st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown(f"**Current Step:** {st.session_state.step} of 4")
st.sidebar.progress(st.session_state.step / 4)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.caption(f"**Index:** {model.index_name}")
st.sidebar.caption(f"**Namespace:** {model.namespace}")
st.sidebar.caption(f"**Embedding:** {model.embedding_model}")
st.sidebar.caption(f"**LLM:** {model.llm_model}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Tech Stack")
st.sidebar.markdown(
    "• LlamaIndex\n\n"
    "• OpenAI Embeddings\n\n"
    "• Pinecone Vector DB\n\n"
    "• GPT-4\n\n"
    "• W&B Weave"
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Demo", use_container_width=True):
    reset_demo()
    st.rerun()
