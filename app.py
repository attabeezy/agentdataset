import os
import streamlit as st
import pandas as pd
import time
from pathlib import Path
from agentdataset.core.orchestrator import Orchestrator

# --- Page Config ---
st.set_page_config(page_title="AgentDataset", page_icon="🤖", layout="wide")

st.title("🤖 AgentDataset: Autonomous Data Factory")
st.markdown("---")

# --- Sidebar: Config ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    model_choice = st.selectbox("LLM Model", ["gpt-4o", "gpt-3.5-turbo"])
    max_iters = st.slider("Max Optimization Loops", 1, 10, 5)
    
    if api_key:
        st.success("API Key loaded")
    else:
        st.warning("Enter API Key to enable LLM extraction")

# --- Session State ---
if "orchestrator" not in st.session_state:
    session_id = f"run_{int(time.time())}"
    st.session_state.orchestrator = Orchestrator(session_id, model=model_choice, api_key=api_key)
    st.session_state.discovery_results = []
    st.session_state.best_data = None

# --- Main: Phase 0 (Discovery) ---
query = st.text_input("What would you like to research? (e.g. 'SME lending in Kenya')", key="search_query")

if st.button("Search Knowledge Sources"):
    with st.spinner("Agent searching web..."):
        results = st.session_state.orchestrator.run_discovery(query)
        st.session_state.discovery_results = results
        st.success(f"Found {len(results)} potential sources.")

if st.session_state.discovery_results:
    st.subheader("Discovered Sources")
    selected_indices = []
    for i, res in enumerate(st.session_state.discovery_results):
        cols = st.columns([0.1, 0.7, 0.2])
        if cols[0].checkbox("Include", value=(i==0), key=f"check_{i}"):
            selected_indices.append(i)
        cols[1].markdown(f"**[{res.title}]({res.url})**")
        cols[2].text(f"Score: {res.relevance_score}")

    if st.button("Generate Dataset from Selected"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extraction
        status_text.text("Extracting statistical DNA...")
        selected_sources = [st.session_state.discovery_results[i] for i in selected_indices]
        
        # For simplicity, we process the first selected source
        if selected_sources:
            params = st.session_state.orchestrator.process_source(selected_sources[0])
            progress_bar.progress(20)
            
            # Step 2: Optimization Loop
            status_text.text("Running Synthesis-Validation Loop...")
            best_score, best_data = st.session_state.orchestrator.run_optimization_loop(params, iterations=max_iters)
            st.session_state.best_data = best_data
            progress_bar.progress(100)
            
            st.success(f"Dataset finalized with Fidelity Score: {best_score}")

# --- Results Panel ---
if st.session_state.best_data is not None:
    st.markdown("---")
    st.subheader("Final Synthetic Dataset")
    st.dataframe(st.session_state.best_data.head(10))
    
    col1, col2 = st.columns(2)
    csv = st.session_state.best_data.to_csv(index=False).encode('utf-8')
    col1.download_button(
        "Download data.csv",
        csv,
        "agentdataset_output.csv",
        "text/csv",
        key='download-csv'
    )
    
    if st.button("Show Distribution Analysis"):
        st.bar_chart(st.session_state.best_data.describe().T['mean'])
