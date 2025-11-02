import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import gc

import streamlit as st
import cleanup_and_trim
import file_utils

# import importlib # for hot reload (dev)
# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Optional Google GenAI (kept optional)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import rag_utils as ru
    HAS_RAG_UTILS = True
    logging.info("rag_utils imported successfully.")
    # importlib.reload(ru) # for hot reload # 🔁 ensures latest tweaks always apply
except Exception:
    ru = None
    HAS_RAG_UTILS = False
    logging.warning("rag_utils not found. Ingest/query buttons will show errors.")


# -- Page config --
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="💬",
    #layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "AI-powered RAG Assistant"
    }
)

# Get the folder where app.py is located
# app_path = Path(__file__).parent
app_path = Path(__file__).resolve().parent

if "reranker_keep_loaded" not in st.session_state:
    st.session_state.reranker_keep_loaded = True
if "ingest_docx" not in st.session_state:
    st.session_state.ingest_docx = False
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if __name__ == "__main__":
    # --- Keep the original sidebar (user asked not to change it) ---
    with st.sidebar:
        st.title("RAG Controls")
        # use_llm = st.checkbox("Use LLM (optional)", value=False)
        # st.session_state.reranker_keep_loaded = st.checkbox(
        #     "Keep reranker loaded [Experimental]", value=st.session_state.reranker_keep_loaded, help="⚡Keeps reranker in memory for faster rerank queries. Disable to save memory."
        # )
        #
        # if st.button("🧹 Unload Reranker Model"):
        #     ru._CE_CACHE.clear()
        #     import gc, torch
        #
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     gc.collect()
        #     st.success("✅ Reranker model unloaded from memory.")

        st.markdown("---")
        st.header("Database Controls")
        if st.button("🗑️ Reset Database"):
            st.session_state.confirm_reset = True

        if st.session_state.confirm_reset:
            st.warning("⚠️ This will permanently delete all documents in your ChromaDB collection!")
            if st.button("Confirm Reset"):
                if not HAS_RAG_UTILS:
                    st.error("rag_utils.py not found — cannot reset collection.")
                    logging.error("Reset failed: rag_utils.py not found.")
                else:
                    try:
                        col = ru.get_collection()
                        # Fetch all documents' metadata to get their IDs
                        all_docs = col.get(include=["metadatas", "documents"])
                        all_ids = [meta.get("id") for meta in all_docs["metadatas"] if "id" in meta]

                        if all_ids:
                            col.delete(ids=all_ids)
                            st.success(f"✅ ChromaDB collection cleared. {len(all_ids)} documents removed.")
                            logging.info(f"ChromaDB collection cleared. {len(all_ids)} documents removed.")
                        else:
                            st.info("Collection is already empty.")

                    except Exception as e:
                        st.exception(e)
                        logging.exception("Failed to reset ChromaDB collection.")
                st.session_state.confirm_reset = False  # reset the confirmation state

            if st.button("Cancel"):
                st.session_state.confirm_reset = False  # reset the confirmation state
                st.rerun()

        st.markdown("---")
        st.header("KB / Ingest")
        kb_dir = st.text_input("KB folder (relative)", value="kb", help="Knowledge Base to ingest, defaulted to kb folder in main folder")
        chunk_size = st.number_input("Chunk size (chars)", value=100000, step=1000)
        overlap = st.number_input("Overlap (chars)", value=200, step=50)
        batch_size = st.number_input("Batch size (upsert)", value=64, step=1)

        st.caption("Ingest `.txt` and `.md` files defaulted")

        st.session_state.ingest_docx = st.checkbox(
            "docx", value=st.session_state.ingest_docx, help="Ingest .docx files -> read Docx files and convert into MD files then ingest to collection."
        )

        if st.button("Ingest KB", help="Ingest KB files and insert into collection for retrieval."):
            logging.info("Ingest button clicked.")
            if not HAS_RAG_UTILS:
                st.error(
                    "rag_utils.py not found. Save your ingestion functions as rag_utils.py or place the helper functions here.")
                logging.error("Ingest failed: rag_utils.py not found.")
            else:
                st.info("Starting ingestion... this runs in the app process — it may take time for large KBs.")
                logging.info(
                    f"Starting ingestion: kb_dir={kb_dir}, chunk_size={chunk_size}, overlap={overlap}, batch_size={batch_size}")
                try:
                    kb_path = app_path / kb_dir if kb_dir == "kb" else Path(kb_dir)
                    logging.info(f"kb_path = {kb_path}")
                    kb_path.mkdir(exist_ok=True)
                    count = ru.ingest_kb_to_collection(
                        app_dir=app_path,
                        kb_dir=kb_path,
                        ingest_docx_flag=st.session_state.ingest_docx,
                        collection=ru.get_collection(),
                        chunk_size=chunk_size,
                        overlap=overlap,
                        batch_size=batch_size,
                        file_globs=["**/*.md", "**/*.txt"], # Dev note: remember to edit .query
                        # file_globs=["**/*.txt"],
                    )
                    st.success(f"Ingestion finished: {count} chunks upserted.")
                    logging.info(f"Ingestion finished: {count} chunks upserted.")
                except Exception as e:
                    st.exception(e)
                    logging.exception("Error during ingestion.")

        st.markdown("---")
        st.write("Notes:")
        st.caption("🗑️ **Reset Database:**\n"
            "Delete the contents in the `chromadb/` folder.\n")
        st.caption(
            "🧠 **Developer Notes:**\n"
            "This module implements core RAG (Retrieval-Augmented Generation) utilities including:\n"
            "- Knowledge base ingestion and chunking\n"
            "- ChromaDB persistence and querying\n"
            "- Cross-encoder reranking for relevance scoring\n"
            "- Optional LLM-based answer synthesis via Gemini API\n"
            "- May tweak the prompt for llm in `rag_utils.py`\n"
        )

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts {role, text, ts}
    if "results" not in st.session_state:
        st.session_state.results = []  # retrieval results
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = False

    # Input area at the bottom (chat-like)
    with st.form(key="chat_form", clear_on_submit=False):
        input_col, btn_col = st.columns([7, 3])

        with input_col:
            user_input = st.text_area(
                "Message",
                value="",
                placeholder="Type your message here and press Send — Shift+Enter for newline",
                height=70,
                key="chat_input",
            )

            # For convenient (horizontal)
            # ✅ Place checkboxes side by side below the text area
            cb1, cb2 = st.columns(2)
            with cb1:
                st.session_state.use_llm = st.checkbox("Use LLM (optional)", value=False)
            with cb2:
                st.session_state.rerank_flag = st.checkbox("Use rerank (optional)", value=True)

            # # For convenient (vertical)
            # st.session_state.use_llm = st.checkbox("Use LLM (optional)", value=False)
            # st.session_state.rerank_flag = st.checkbox("Use rerank (optional)", value=True)


        with btn_col:
            st.write("")  # adds small space
            st.write("")  # add another if you want lower
            # create 3 equal-width columns inside the right panel
            b1, b2 = st.columns(2)
            with b1:
                send = st.form_submit_button("Send")
            with b2:
                clear = st.form_submit_button("Clear")

            # handle clear button
            if clear:
                st.session_state.messages = []
                st.session_state.results = []
                st.rerun()

        if send and user_input.strip():
            logging.info(f"User submitted: {user_input.strip()}")
            # throw early if dont have rag_utils.py
            if not HAS_RAG_UTILS:
                raise ImportError("RAG utilities module not found — please ensure rag_utils.py is available.")

            if HAS_RAG_UTILS:
                try:
                    logging.info("Querying collection...")
                    results = ru.get_collection().query(
                        query_texts=user_input,
                        n_results=10,
                        where={"type": {"$in": ["md", "txt","docx"]}},
                        include=["documents", "metadatas", "distances"],
                    )
                    logging.info("Query successful.")

                    # Transform results to list[dict[str, Any]] - easier access data, supports customization
                    top_records = ru.transform_result(results)

                except Exception as e:
                    logging.exception("Error while querying collection.")
                    st.session_state.messages.append(
                        {'role': 'bot', 'text': f"**Error querying collection:** {e}", 'ts': time.time()})
                    results = None

                rerank_flag = st.session_state.rerank_flag
                rerank_top_records: List[Dict[str, Any]] = []

                logging.info(f"rerank_flag: {st.session_state.rerank_flag}")
                if st.session_state.rerank_flag:
                    # rerank
                    logging.info("Re-ranking...")
                    # rerank_pairs: list[tuple[dict[str, Any], float]]
                    rerank_pairs = ru.rerank_with_cross_encoder_v2(user_input, top_records, stay_active=st.session_state.reranker_keep_loaded) if top_records else []
                    # transform rerank_pairs to List[Dict[str, Any]] = []
                    for rank, (rec, score) in enumerate(rerank_pairs, start=1):
                        rerank_top_records.append(rec)
                        # for debugging
                        # print(f"{rank}. [{score:.3f}] {rec.get('title', '?')} | {rec.get('type', '?')}")
                        # print(f"    note: {rec.get('title')} | {rec.get('snippet')}")
                    logging.info("Re-ranking... complete.")

                # Format results into st.session_state.results
                # Choose which list to display (reranked if available)
                chosen_records = rerank_top_records if rerank_top_records else top_records or []

                if rerank_top_records:
                    logging.info(f"Using reranked results ({len(rerank_top_records)} records).")
                elif top_records:
                    logging.info(f"Using original top_records ({len(top_records)} records).")
                else:
                    logging.info("No records found (both reranked and top_records are empty).")

                # Populate the UI results from chosen_records
                st.session_state.results = []
                for i, rec in enumerate(chosen_records, start=1):
                    doc = rec.get("document", "") or ""
                    meta = rec.get("metadata", {}) or {}
                    # Prefer explicit similarity if present
                    sim = rec.get("similarity", None)
                    if sim is None:
                        # Try to infer from `score`:
                        # - if transform_result stored `score` as distance, similarity = 1 - distance
                        # - if score looks like already a similarity (0..1) then use it directly
                        score = rec.get("score", None)
                        if isinstance(score, (int, float)):
                            if 0.0 <= score <= 1.0:
                                # heuristics: if score likely a distance (close to 1 => far) convert,
                                # but we can't be sure, so use the following conservative approach:
                                # if transform_result stored distance (your code did `score=float(d)`),
                                # convert to similarity = 1 - distance
                                sim = 1.0 - float(score)
                            else:
                                # score outside [0,1] — treat as unavailable
                                sim = None
                        else:
                            sim = None

                    snippet = (" ".join(doc.split()))[:140]
                    if len(" ".join(doc.split())) > 140:
                        snippet = snippet.rstrip() + "..."

                    entry = {
                        "rank": i,
                        "similarity": float(sim) if sim is not None else None,
                        "title": meta.get("title") or Path(meta.get("source_file", "")).stem,
                        "source": meta.get("source_file") or meta.get("source") or "",
                        "snippet": snippet,
                        "document": doc,
                        "metadata": meta,
                    }
                    st.session_state.results.append(entry)

            # response = call_llm(user_input, top_records[:10]) if top_records else ""

            if st.session_state.use_llm and HAS_RAG_UTILS:
                try:
                    logging.info("# Sending to LLM")
                    response = ru.call_llm(question=user_input, top_records=chosen_records[:10], stream=True)
                    # st.write(response)
                    st.markdown("---")
                    st.markdown("## 💬")
                    st.write_stream(response)
                    st.markdown("---")

                except Exception as e:
                    logging.exception("call_llm failed")
                    response = f"(LLM call failed: {e})"
            else:
                # no LLM path: use top snippet or a no-result message
                if st.session_state.results:
                    top = st.session_state.results[0]
                    response = f"**Top snippet from** {top['title']}\n\n{top['snippet']}"
                else:
                    response = "No results found in the vector DB."

            # logging.info(f"st.session_state.results: {st.session_state.results}" )

            # clear memory
            try:
                del results, top_records, rerank_pairs, rerank_top_records
            except Exception:
                pass
            gc.collect()
            
            st.markdown("## 📚 Sources")

            # logging.info(st.session_state.results)


        # List out documents
        for i, r in enumerate(st.session_state.results):
            col1, col2, col3, col4 = st.columns([6, 1, 2.7, 2.5])  # adjust ratios as needed
            with col1:
                st.markdown(f"**{r['rank']}. {r['title']}**")
            with col3:
                if st.form_submit_button("📂 Open folder", key=f"open_folder_{i}"):
                    metadata = r.get("metadata", {})
                    folder = metadata.get("folder")                             # example: C:\Users\xxxx\
                    logging.info(f"📂 Opening folder... : folder={folder}")
                    file_utils.open_folder(folder)
            with col4:
                if st.form_submit_button("📄Open file", key=f"open_file_{i}"):
                    metadata = r.get("metadata", {})
                    source_file_full = metadata.get("source_file_full")         # example: C:\Users\xxxx\document.txt
                    logging.info(f"📄 Opening file... : source_file_full={source_file_full}")
                    file_utils.open_file(source_file_full)

            sim = r.get('similarity')
            # form_key = f"chat_form_{i}"
            # st.caption(f"Similarity: {sim:.4f}" if sim is not None else "Similarity: N/A")
            # st.caption(f"Source: {r['source']}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(f"Similarity: {sim:.4f}" if sim is not None else "Similarity: N/A")
            with col2:
                st.caption(f"Source: {r['source']}")

            with st.expander("View snippet"):
                st.text(r['snippet'])

            with st.expander("Full document"):
                st.code(r['document'], language="text")

            with st.expander("Metadata"):
                st.json(r['metadata'])

            st.markdown("---")

        with st.expander("See retrieved documents"):
            st.write(st.session_state.results)

        with st.expander("See most relevant document ids"):
            st.write("test")
            st.write("relevant_text")

        # Clean up, free memory
        cleanup_and_trim.best_effort_idle_release()


