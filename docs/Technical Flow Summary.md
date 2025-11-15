# 🧠 RAG App — Technical Flow Specification

This document explains the internal data flow and logic sequence of the **RAG (Retrieval-Augmented Generation)** Streamlit app.
The system integrates document ingestion, vector retrieval, re-ranking, and optional LLM response generation.


## Summary ##
User Input → Query Vector DB → Retrieve Top N → Rerank → (Optional) Prompt LLM → Response
         ↑                     ↓
      Ingest KB           Context Display



---

## ⚙️ Overall Structure

**Main app file:** `app_main.py`
**Helper module:** `rag_utils.py`
**Framework:** Streamlit
**Key dependencies:** `chromadb`, `google.generativeai` (optional), `cross-encoder`, `sentence-transformers`, etc.

---

## 1. 🗂 Ingest Flow

### Purpose

Convert source documents (`.txt`, `.md`) into vector embeddings and upsert them into a **ChromaDB collection** for semantic retrieval.

### Trigger

User clicks **“Ingest KB”** in the **sidebar**.

### Flow

1. Read user inputs:

   * `kb_dir` → target directory for knowledge base files.
   * `chunk_size`, `overlap` → control text segmentation.
   * `batch_size` → batch size for upserts.

2. Call:

   ```python
   count = ru.ingest_kb_to_collection(
       app_dir=app_path,
       kb_dir=Path(kb_dir),
       collection=ru.get_collection(),
       chunk_size=chunk_size,
       overlap=overlap,
       batch_size=batch_size,
       file_globs=["**/*.md", "**/*.txt"]
   )
   ```

3. **`ingest_kb_to_collection()`**:

   * Reads and splits documents into chunks.
   * Embeds chunks using the selected embedding model.
   * Stores chunks + metadata into the persistent ChromaDB collection.

4. Logs ingestion progress and reports the number of chunks upserted.

---

## 2. 🔍 Query Flow

### Purpose

Retrieve the most semantically relevant document chunks from the vector store.

### Trigger

User submits a message in the main input area and clicks **“Send”**.

### Flow

1. Extract query text:

   ```python
   user_input = st.session_state.chat_input
   ```

2. Call ChromaDB query:

   ```python
   results = ru.get_collection().query(
       query_texts=user_input,
       n_results=10,
       where={"type": {"$in": ["md", "txt"]}},
       include=["documents", "metadatas", "distances"]
   )
   ```

3. Transform query results into a list of structured dictionaries:

   ```python
   top_records = ru.transform_result(results)
   ```

4. Populate `st.session_state.results` with fields:

   * `rank`
   * `title`
   * `similarity`
   * `source`
   * `snippet`
   * `document`
   * `metadata`

---

## 3. 🧩 Rerank Flow

### Purpose

Improve ranking accuracy using a **Cross-Encoder** model that scores query–document pairs.

### Flow

1. Send top retrieved records to:

   ```python
   rerank_pairs = ru.rerank_with_cross_encoder(user_input, top_records)
   ```
2. Each pair `(record, score)` is sorted by descending score.
3. Extract top-ranked records:

   ```python
   rerank_top_records = [rec for rec, _ in rerank_pairs]
   ```
4. Update session with re-ranked list for downstream use.

---

## 4. ✍️ Prompt Creation

### Purpose

Construct a contextual prompt for the LLM using the user’s query and top documents.

### Flow

When `use_llm` is **enabled**:

1. Call:

   ```python
   response = ru.call_llm(
       question=user_input,
       top_records=rerank_top_records[:10],
       stream=True
   )
   ```
2. Inside `call_llm()`:

   * Combine user question and the top retrieved context chunks.
   * Create a formatted text prompt, e.g.:

     ```
     Question: {user_input}
     Context:
     1. {doc1_snippet}
     2. {doc2_snippet}
     ...
     ```

---

## 5. 🧠 Feed LLM

### Purpose

Send the composed prompt to the LLM (e.g., Google Gemini, OpenAI GPT, etc.) to generate a synthesized response.

### Flow

1. `call_llm()` invokes the chosen LLM API.

2. The model processes the prompt with context.

3. Streamlit displays the live response stream using:

   ```python
   st.write_stream(response)
   ```

4. Any error during model execution is caught and logged.

---

## 6. 💬 Response Display

### Purpose

Show retrieved context and model outputs in a user-friendly format.

### Flow

1. If LLM is **enabled**:

   * The generated answer is displayed in the main chat area.

2. If LLM is **disabled**:

   * The system displays a “Top snippet” extracted from the most similar document:

     ```python
     response = f"**Top snippet from** {top['title']}\n\n{top['snippet']}"
     ```

3. Below the response, retrieval details are rendered:

   * Document list with similarity scores.
   * Expanders for snippet, full document, and metadata.
   * Optional raw query results.

---

## 🧭 Summary of Data Flow

```
User Input → Query Vector DB → Retrieve Top N → Rerank → (Optional) Prompt LLM → Response
         ↑                     ↓
      Ingest KB           Context Display
```

---

## 🪵 Logging and Debugging

* Logging via Python’s built-in `logging` module:

  * `INFO`: standard flow (ingestion, query, rerank)
  * `ERROR`: exception during operations
* Logs appear in the Streamlit terminal.

---

## 🔐 Notes and Best Practices

* Ensure `rag_utils.py` exports:

  * `get_collection()`
  * `ingest_kb_to_collection()`
  * `transform_result()`
  * `rerank_with_cross_encoder()`
  * `call_llm()`

* Use consistent chunking and embedding configuration during ingestion and retrieval.

* For large KBs, ingestion may be CPU-intensive — consider running offline or asynchronously.