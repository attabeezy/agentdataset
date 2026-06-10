import os
import time
import streamlit as st
import pandas as pd
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.discovery import SearchError

# --- Provider config ---
PROVIDERS = {
    "OpenAI": {
        "env_var": "OPENAI_API_KEY",
        "key_label": "OpenAI API Key",
        "models": ["gpt-4o", "gpt-3.5-turbo"],
    },
    "Claude": {
        "env_var": "ANTHROPIC_API_KEY",
        "key_label": "Anthropic API Key",
        "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    },
    "Gemini": {
        "env_var": "GEMINI_API_KEY",
        "key_label": "Google Gemini API Key",
        "models": ["gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"],
    },
}

# --- Page Config ---
st.set_page_config(page_title="AgentDataset", page_icon="🤖", layout="wide")
st.title("🤖 AgentDataset: Autonomous Data Factory")
st.markdown("---")

# --- Sidebar: Config ---
with st.sidebar:
    st.header("Settings")

    provider_choice = st.selectbox("API Provider", list(PROVIDERS.keys()))
    cfg = PROVIDERS[provider_choice]

    api_key = st.text_input(
        cfg["key_label"],
        value=os.environ.get(cfg["env_var"], ""),
        type="password",
    )
    model_choice = st.selectbox("Model", cfg["models"])
    max_iters = st.slider("Max Optimization Loops", 1, 10, 5)

    if api_key:
        st.success("API Key loaded")
    else:
        st.warning("Enter API Key to enable LLM extraction")

# --- Session State ---
# Recreate the orchestrator whenever provider or model changes
config_key = (provider_choice, model_choice, api_key)
if (
    "orchestrator" not in st.session_state
    or st.session_state.get("config_key") != config_key
):
    session_id = f"run_{int(time.time())}"
    st.session_state.orchestrator = Orchestrator(
        session_id,
        model=model_choice,
        api_key=api_key,
        env_var=cfg["env_var"],
    )
    st.session_state.config_key = config_key
    st.session_state.discovery_results = []
    st.session_state.suggested_indices = []
    st.session_state.best_data = None

# --- Main: Phase 0 (Discovery) ---
query = st.text_input(
    "What would you like to research? (e.g. 'SME lending in Kenya')", key="search_query"
)

if st.button("Search Knowledge Sources"):
    if not query or not query.strip():
        st.warning("Please enter a research query before searching.")
    else:
        with st.spinner("Agent optimizing query and searching web..."):
            try:
                results = st.session_state.orchestrator.run_discovery(query.strip())
                st.session_state.discovery_results = results

                # Get suggestions from the agent
                if results:
                    st.session_state.suggested_indices = (
                        st.session_state.orchestrator.suggest_sources(results)
                    )
                else:
                    st.session_state.suggested_indices = []

                if results:
                    st.success(f"Found {len(results)} potential sources.")
                else:
                    st.info(
                        "No sources found for that query. Try different or broader terms."
                    )
            except SearchError as e:
                st.session_state.discovery_results = []
                st.session_state.suggested_indices = []
                st.error(f"Search failed (the search backend returned an error): {e}")

if st.session_state.discovery_results:
    st.subheader("Discovered Sources")

    # Add a button to automatically select suggested sources
    suggested = st.session_state.get("suggested_indices", [])
    if suggested:
        if st.button("✨ Select Suggested Sources"):
            # We can't directly mutate the checkboxes from here without a re-run logic,
            # so we'll store the "auto-select" intent in session state.
            st.session_state.auto_select_suggested = True
            st.rerun()

    selected_indices = []
    for i, res in enumerate(st.session_state.discovery_results):
        cols = st.columns([0.1, 0.7, 0.2])

        # Determine default value: if auto-select is on, use suggested indices
        is_suggested = i in suggested
        default_val = (
            is_suggested if st.session_state.get("auto_select_suggested") else (i == 0)
        )

        if cols[0].checkbox("Include", value=default_val, key=f"check_{i}"):
            selected_indices.append(i)

        title_text = f"**[{res.title}]({res.url})**"
        if is_suggested:
            title_text += " ✨ (Suggested)"

        cols[1].markdown(title_text)
        cols[2].text(f"Score: {res.relevance_score}")

    # Reset auto-select flag after rendering checkboxes to avoid perpetual selection
    if "auto_select_suggested" in st.session_state:
        del st.session_state.auto_select_suggested

    if st.button("Generate Dataset from Selected"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        selected_sources = [
            st.session_state.discovery_results[i] for i in selected_indices
        ]

        if not selected_sources:
            progress_bar.empty()
            status_text.empty()
            st.warning("Please select at least one source to generate a dataset.")
        else:
            # Extract parameters from every selected source
            all_params = []
            for idx, source in enumerate(selected_sources):
                status_text.text(
                    f"Extracting statistical DNA from source {idx + 1}/{len(selected_sources)}..."
                )
                all_params.append(st.session_state.orchestrator.process_source(source))
                progress_bar.progress(int(20 * (idx + 1) / len(selected_sources)))

            # Merge if multiple sources were selected
            params = st.session_state.orchestrator.merge_parameters(all_params)
            if len(selected_sources) > 1:
                status_text.text(
                    f"Merged parameters from {len(selected_sources)} sources."
                )

            progress_bar.progress(20)

            # Guard: extraction produced no usable variables → stop with a clear message
            if not params.variables:
                progress_bar.empty()
                status_text.empty()
                methods = ", ".join(
                    sorted({p.meta.extraction_method for p in all_params})
                )
                st.error(
                    "**No statistical parameters could be extracted from the selected "
                    "source(s), so no dataset was generated.**\n\n"
                    "This usually means one of the following:\n"
                    "- The source PDF could not be downloaded (some hosts block automated "
                    "access with a 403), so only a short search snippet was available.\n"
                    "- LLM extraction was unavailable (no API key, or the provider returned "
                    "a rate-limit / quota error) and the regex fallback found no mean/std pairs.\n\n"
                    f"Extraction method used: `{methods}`.\n\n"
                    "Try a different source, add a working API key in the sidebar, or pick a "
                    "source whose document is directly downloadable."
                )
            else:
                status_text.text("Running Synthesis-Validation Loop...")
                best_score, best_data = (
                    st.session_state.orchestrator.run_optimization_loop(
                        params, iterations=max_iters
                    )
                )
                st.session_state.best_data = best_data
                progress_bar.progress(100)

                st.success(f"Dataset finalized with Fidelity Score: {best_score}")

# --- Results Panel ---
if st.session_state.best_data is not None and not st.session_state.best_data.empty:
    st.markdown("---")
    st.subheader("Final Synthetic Dataset")
    st.dataframe(st.session_state.best_data.head(10))

    col1, col2 = st.columns(2)
    csv = st.session_state.best_data.to_csv(index=False).encode("utf-8")
    col1.download_button(
        "Download data.csv",
        csv,
        "agentdataset_output.csv",
        "text/csv",
        key="download-csv",
    )

    if st.button("Show Distribution Analysis"):
        st.bar_chart(st.session_state.best_data.describe().T["mean"])
