# rag_utils.py
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from pathlib import Path
import os, re, json, hashlib, time
from typing import List, Dict, Any, Tuple, Optional
import math
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import logging
import nltk   # optional: for sentence tokenization if available
import gc

try:
    import convert_docx_to_markdown as docx2md
except Exception:
    docx2md = None

try:
    import torch
except Exception:
    torch = None

# Optional Google GenAI (kept optional)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ----------------- Common Path -----------------
# Get the folder where app.py is located
app_path = Path(__file__).parent
# Define nested folder path: app_path/Ingestion/docx
ingestion_docx_path = app_path / "Ingestion" / "docx"  # C:\Users\rag_minimal\Ingestion\docx\

def call_llm(question: str, top_records: List[Dict[str, Any]], stream=True):

    logging.info("Loading key....")
    load_dotenv()
    api_key = os.getenv("GOOGLE_GENAI_API_KEY", "") or os.getenv("GENAI_API_KEY", "")
    context = "\n\n".join(f"[{r.get('id')}], Title: [{r.get('title')}]\n {r.get('document')}" for r in top_records)
    logging.info("Key loaded successful....")
    if not api_key or genai is None:
        return ""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

        context will be passed as "Context:"
        user question will be passed as "Question:"

        To answer the question:
        1. Thoroughly analyze the context, identifying key information relevant to the question.
        2. Organize your thoughts and plan your response to ensure a logical flow of information.
        3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
        4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
        5. Always use technical-style operators. (eg, :, >, <, <>, <=, >=, arrows)
        6. Always show tables if possible.
        7. If the context doesn't contain sufficient information to fully answer the question, state this clearly early in your response.

        Format your response as follows:
        1. Use clear, concise language.
        2. Organize your answer into paragraphs for readability, markdown format.
        3. Use bullet points or numbered lists or tables where appropriate to break down complex information.
        4. Beautify the response with emojis.
        5. Ensure proper grammar, punctuation, and spelling throughout your answer.
        6. Always produce output that is **easy for a human to read and understand**, similar to a ledger or scenario report.
                
        Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.

        Context: {context}
        Question: {question}

        """

        if stream:
            logging.info("response_stream")
            # Get stream of chunks
            response_stream = model.generate_content(contents=prompt,stream=True)

            # Yield text chunks as they arrive
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                elif isinstance(chunk, dict) and "text" in chunk:
                    yield chunk["text"]
        else:
            logging.info("response")
            response = model.generate_content(contents=prompt, stream=False)
            return response



    except Exception as e:
        yield f"⚠️ Error during streaming: {e}"
        return f"(LLM disabled or error: {e})"


# ----------------- Helpers & LLM (unchanged mostly) -----------------
def maybe_answer_with_llm(question: str, top_records: List[Dict[str, Any]]):

    logging.info("Loading key....")
    load_dotenv()
    api_key = os.getenv("GOOGLE_GENAI_API_KEY", "") or os.getenv("GENAI_API_KEY", "")
    logging.info("Key loaded successful....")
    if not api_key or genai is None:
        return ""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        context = "\n\n".join(f"[{r.get('id')}], Title: [{r.get('title')}]\n {r.get('document')}" for r in top_records)
        logging.info(context)

        prompt = f"""
        You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

        context will be passed as "Context:"
        user question will be passed as "Question:"

        To answer the question:
        1. Thoroughly analyze the context, identifying key information relevant to the question.
        2. Organize your thoughts and plan your response to ensure a logical flow of information.
        3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
        4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
        5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

        Format your response as follows:
        1. Use clear, concise language.
        2. Organize your answer into paragraphs for readability.
        3. Use bullet points or numbered lists where appropriate to break down complex information.
        4. If relevant, include any headings or subheadings to structure your response.
        5. Ensure proper grammar, punctuation, and spelling throughout your answer.

        Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.

        Context: {context}
        Question: {question}

        """

        response_stream = model.generate_content(
            contents=prompt,
            stream=True
        )

        # Yield text chunks as they arrive
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
            elif isinstance(chunk, dict) and "text" in chunk:
                yield chunk["text"]

        # return getattr(response_stream, "text", "").strip()
    except Exception as e:
        yield f"⚠️ Error during streaming: {e}"
        return f"(LLM disabled or error: {e})"


# Cross encoder cache
_CE_CACHE: Dict[str, CrossEncoder] = {}
def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    if model_name not in _CE_CACHE:
        _CE_CACHE[model_name] = CrossEncoder(model_name)
    return _CE_CACHE[model_name]

# Unload cross encoder. (Release Memory)
def unload_cross_encoder(model_name: str):
    """Unload a specific CrossEncoder model from the cache and free its memory."""
    if model_name in _CE_CACHE:
        model = _CE_CACHE.pop(model_name)

        # Explicitly delete model attributes (important for GPU/torch models)
        if hasattr(model, 'model'):
            del model.model
        if hasattr(model, 'tokenizer'):
            del model.tokenizer

        # Clear CUDA memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python garbage collection
        gc.collect()
        print(f"✅ Unloaded model: {model_name}")
    else:
        print(f"⚠️ Model not found in cache: {model_name}")

def rerank_with_cross_encoder(
    query: str,
    records: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
    max_passage_chars: int = 1500,
) -> List[Tuple[Dict[str, Any], float]]:
    ce = _get_cross_encoder(model_name)
    pairs = []
    for rec in records:
        passage = rec["document"]
        if len(passage) > max_passage_chars:
            passage = passage[:max_passage_chars]
        pairs.append((query, passage))
    ce_scores = ce.predict(pairs, batch_size=batch_size)
    order = np.argsort(-np.asarray(ce_scores))
    return [(records[i], float(ce_scores[i])) for i in order]

def _truncate_to_sentence(text: str, max_chars: int) -> str:
    """Try to truncate at sentence boundary; fallback to hard cut."""
    if len(text) <= max_chars:
        return text
    # best-effort: use simple sentence split if nltk is available
    try:
        sents = nltk.tokenize.sent_tokenize(text)
        out = ""
        for s in sents:
            if len(out) + len(s) + 1 > max_chars:
                break
            out = (out + " " + s).strip()
        return out if out else text[:max_chars]
    except Exception:
        return text[:max_chars]

# helper: try sentence/paragraph split, fallback to char chunks
def _smart_chunks(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    try:
        import nltk
        sents = nltk.tokenize.sent_tokenize(text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                # if single sentence > max_chars, hard-split the sentence
                if len(s) > max_chars:
                    for i in range(0, len(s), max_chars):
                        chunks.append(s[i:i + max_chars])
                    cur = ""
                else:
                    cur = s
        if cur:
            chunks.append(cur)

        # ✅ Clean up large variables explicitly
        sents = None
        cur = None

        import gc
        gc.collect()

        return chunks

    except Exception:
        out = []
        start = 0
        while start < len(text):
            out.append(text[start:start + max_chars])
            start += max_chars

        # ✅ Clean up large variables
        text = None
        start = None

        import gc
        gc.collect()

        return out

def _make_pair(query: str, passage: str) -> Tuple[str, str]:
    # cross-encoder expects (query, passage) or similar
    return (query, passage)

def rerank_with_cross_encoder_v2(
    query: str,
    records: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
    max_passage_chars: int = 1500,
    top_k: Optional[int] = None,
    chunk_long_docs: bool = True,
    agg_strategy: str = "max",                 # "max" | "avg" | "length_weighted"
    blend_retriever_score: float = 0.0,        # weight for retriever score in final scoring (0.0 = ignore)
    reranker_score_weight: float = 1.0,        # weight for reranker score
    ce_predict_fn = None,                      # allows injecting custom CE predict function for testing
    stay_active = False,
) -> List[Tuple[Dict[str, Any], float]]:
    if not records:
        logging.debug("rerank_with_cross_encoder_v2 called with no records")
        return []

    # load cross-encoder, or use injected function
    if ce_predict_fn is None:
        ce = _get_cross_encoder(model_name)  # user-provided loader
        # wrap to a predict function
        def _predict(pairs, batch_size_local=batch_size):
            try:
                return ce.predict(pairs, batch_size=batch_size_local)
            except Exception as e:
                logging.exception("Cross-encoder prediction failed")
                raise
        predict_fn = _predict
    else:
        predict_fn = ce_predict_fn
        ce = None  # no local CE instance when using injected fn

    # build pairs and mapping: pair_idx -> record_idx, also keep lengths for weighting
    pairs = []
    mapping = []
    chunk_lens = []

    for i, rec in enumerate(records):
        title = rec.get("title", "")
        passage = rec.get("document", "") or ""
        # combine title + passage but keep title small:
        combined = (title + "\n\n" + passage).strip() if title else passage

        if chunk_long_docs:
            chunks = _smart_chunks(combined, max_passage_chars)
        else:
            truncated = combined[:max_passage_chars]
            chunks = [truncated] if truncated else []

        if not chunks:
            chunks = [""]  # safely ask model to score empty if needed

        for c in chunks:
            pairs.append(_make_pair(query, c))
            mapping.append(i)
            chunk_lens.append(len(c))

    if not pairs:
        logging.debug("No pairs were created for reranking")
        return []

    # run prediction in batches — predict_fn should accept list of (query, passage) pairs
    ce_scores = predict_fn(pairs, batch_size)

    # ensure numpy array
    ce_scores = np.asarray(ce_scores, dtype=float)
    if ce_scores.shape[0] != len(pairs):
        raise RuntimeError(f"ce.predict returned {ce_scores.shape[0]} scores but expected {len(pairs)}")

    # aggregate chunk scores back into record-level lists
    temp = {}
    temp_lens = {}
    for idx, s in enumerate(ce_scores):
        rec_idx = mapping[idx]
        temp.setdefault(rec_idx, []).append(float(s))
        temp_lens.setdefault(rec_idx, []).append(int(chunk_lens[idx]))

    # compute aggregated reranker score per record
    reranker_scores = {}
    for rec_idx, scores in temp.items():
        if agg_strategy == "max":
            agg = max(scores)
        elif agg_strategy == "avg":
            agg = float(sum(scores)) / len(scores)
        elif agg_strategy == "length_weighted":
            lens = temp_lens[rec_idx]
            weighted_sum = sum(s * l for s, l in zip(scores, lens))
            agg = float(weighted_sum / sum(lens)) if sum(lens) > 0 else float(sum(scores) / len(scores))
        else:
            raise ValueError(f"Unknown agg_strategy: {agg_strategy}")
        reranker_scores[rec_idx] = agg

    # optional: normalize reranker scores to 0..1 (min-max) so blending makes sense
    all_r_scores = np.array(list(reranker_scores.values()), dtype=float)
    if all_r_scores.size == 0:
        logging.debug("No reranker scores computed")
        return []

    # min-max norm (if constant, set to zeros)
    r_min, r_max = float(all_r_scores.min()), float(all_r_scores.max())
    normalized_r = {}
    if math.isclose(r_max, r_min):
        for k in reranker_scores:
            normalized_r[k] = 0.5  # neutral middle if all equal
    else:
        for k, v in reranker_scores.items():
            normalized_r[k] = (v - r_min) / (r_max - r_min)

    # handle retriever score if present: collect and normalize (if blending)
    normalized_retriever = {}
    if blend_retriever_score and any("retriever_score" in rec for rec in records):
        retriever_vals = []
        for rec in records:
            val = rec.get("retriever_score")
            retriever_vals.append(float(val) if val is not None else float('nan'))
        # replace nan with min or 0
        retriever_vals = np.array([0.0 if math.isnan(x) else x for x in retriever_vals], dtype=float)
        rv_min, rv_max = float(retriever_vals.min()), float(retriever_vals.max())
        if math.isclose(rv_max, rv_min):
            normalized = [0.5] * len(retriever_vals)
        else:
            normalized = [(x - rv_min) / (rv_max - rv_min) for x in retriever_vals]
        for idx, val in enumerate(normalized):
            normalized_retriever[idx] = float(val)

    # compute final score by blending
    final_scores = {}
    for i in range(len(records)):
        rscore = normalized_r.get(i, 0.0)
        retr_score = normalized_retriever.get(i, 0.0) if blend_retriever_score else 0.0
        final = reranker_score_weight * rscore + blend_retriever_score * retr_score
        final_scores[i] = float(final)

    # create output list, stable sort by score desc, tie-breaker by original index (stable)
    out = [(records[i], final_scores.get(i, 0.0)) for i in range(len(records))]
    out.sort(key=lambda x: (-x[1], records.index(x[0])))

    if top_k is not None:
        out = out[:top_k]

    # logging examples: top chosen items and their scores
    logging.info("Reranker returned %d records (top_k=%s). Top 5:", len(out), str(top_k))
    for rec, sc in out[:5]:
        # log title if exists or first 80 chars of document
        title = rec.get("title") or (rec.get("document", "")[:80].replace("\n", " "))
        logging.info("  score=%.4f  title=%s", sc, title)

    logging.info(f"cross encoder stay_active: {stay_active}")
    if not stay_active:
        unload_cross_encoder(model_name)

    # --- CLEANUP: explicitly drop big local variables and request GC + GPU cache free ---
    try:
        # Set large objects to None so they can be garbage-collected
        pairs = None
        mapping = None
        chunk_lens = None
        temp = None
        temp_lens = None
        reranker_scores = None
        all_r_scores = None
        normalized_r = None
        normalized_retriever = None
        ce_scores = None

        # Drop model/predictor references (important for GPU / torch)
        predict_fn = None
        ce = None
        # If torch is available, try to free CUDA memory (best-effort)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # ignore if torch not installed or any error
            pass
        # force a garbage collection sweep
        gc.collect()
    except Exception:
        # ensure that even if cleanup fails, function still returns result
        logging.exception("Cleanup before return failed")

    return out

# ----------------- Ingestion helpers -----------------
def _clean_text(txt: str) -> str:
    # txt = re.sub(r"\r\n?", "\n", txt)
    # txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _extract_title_from_md(text: str, filename: str) -> str:
    m = re.search(r"^\s{0,3}#{1,6}\s+(.*)$", text, flags=re.MULTILINE)
    if m:
        title = m.group(1).strip()
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"\*\*|__|\*|_|`", "", title)
        return title
    return Path(filename).stem

def _chunk_text(text: str, chunk_size: int = 100000, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def _doc_id(source_file: str, idx: int) -> str:
    key = f"{source_file}::chunk-{idx}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _read_with_fallback(path: Path) -> str:
    """Try UTF-8 then latin-1. Raise last exception if both fail."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def ingest_kb_to_collection(
    app_dir: Path,
    kb_dir: Path,
    ingest_docx_flag=False,
    collection=None,
    chunk_size: int = 100000,
    overlap: int = 200,
    batch_size: int = 64,
    file_globs: List[str] = None,
) -> int:
    if file_globs is None:
        file_globs = ["**/*.md", "**/*.txt"]

    if ingest_docx_flag: # add suffix search to ingest
        file_globs += ["**/*.docx"]

        # INGEST DOCX : Convert .docx to .md before ingest in [rag_minimal/Ingestion/docx] (if applicable)
        docx_to_md(kb_dir)

    files = []
    for g in file_globs:
        files.extend(sorted(kb_dir.glob(g)))

    print(f"Found {len(files)} files to ingest under {kb_dir}")

    documents = []
    metadatas = []
    ids = []
    count_chunks = 0

    dump_path = app_dir / "kb_chunks.jsonl"
    try:
        dump_f = open(dump_path, "w", encoding="utf-8")
        print(f"Writing chunk dump to {dump_path}")
    except Exception as e:
        dump_f = None
        print(f"Warning: could not open {dump_path} for writing: {e}")

    try:
        for fpath in files:
            try:
                text = "" # Initialize text <-- bug prevention, every file should have new ""
                suffix = fpath.suffix.lower()
                match suffix:
                    # ────────────────────────────────
                    # Markdown / Text Files
                    # ────────────────────────────────
                    case ".md" | ".txt":
                        try:
                            text = _read_with_fallback(fpath)
                            logging.info("✅ Read text file: %s", fpath.name)
                        except Exception as e:
                            print(f"Skipping {fpath} due to read error: {e}")
                            continue

                    # ────────────────────────────────
                    # DOCX Files
                    # ────────────────────────────────
                    case ".docx":
                        # Try matching converted filenames
                        candidates = [
                            ingestion_docx_path / f"{fpath.stem}.md",
                            ingestion_docx_path / f"{fpath.stem} [rag].md",
                        ]

                        # Find converted .docx->.md file in ingestion_docx_path
                        found = next((c for c in candidates if c.exists()), None)
                        if not found:
                            logging.warning("No converted file found for %s", fpath.name)
                            continue

                        try:
                            text = _read_with_fallback(found)
                            logging.info("📄 Using converted file: %s -> %s", fpath.name, found.name)
                        except Exception as e:
                            logging.error("Skipping %s (converted %s) due to read error: %s",fpath.name, found.name, e)
                            continue
                    # ────────────────────────────────
                    # Unsupported file types
                    # ────────────────────────────────
                    case _:
                        logging.info("Skipping unsupported file type: %s", fpath)
                        continue
            except Exception as e:
                logging.exception("Unexpected error processing %s: %s", fpath, e)
                continue

            text = _clean_text(text)
            if not text:
                continue

            title = fpath.stem
            chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            file_type = fpath.suffix[1:].lower() or "unknown"  # e.g. ".md" -> "md"
            for idx, chunk in enumerate(chunks):
                doc_id = _doc_id(str(fpath), idx)
                meta = {
                    # "source_file": str(fpath.relative_to(kb_dir)).replace("\\", "/"),
                    "id": doc_id,
                    "source_file": str(Path(kb_dir.name) / fpath.relative_to(kb_dir)),
                    "source_file_full": str(fpath),
                    "folder": str(fpath.parents[0]),
                    "title": title,
                    "chunk_index": idx,
                    "type": file_type,
                }
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append(meta)
                count_chunks += 1

                if dump_f:
                    try:
                        json_obj = {
                            "id": doc_id,
                            # "source_file": str(fpath.relative_to(kb_dir)).replace("\\", "/"),
                            "source_file": str(Path(kb_dir.name) / fpath.relative_to(kb_dir)),
                            "folder": str(fpath.parents[0]),
                            "source_file_full": str(fpath),
                            "title": title,
                            "chunk_index": idx,
                            "text": chunk,
                        }
                        dump_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Warning: failed to write chunk to dump: {e}")

                if len(ids) >= batch_size:
                    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                    print(f"Upserted batch of {len(ids)} chunks. Total so far: {count_chunks}")
                    ids = []
                    documents = []
                    metadatas = []
                    time.sleep(0.01)

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Upserted final batch of {len(ids)} chunks. Total: {count_chunks}")

    finally:
        if dump_f:
            dump_f.flush()
            dump_f.close()

    return count_chunks

def docx_to_md(kb_path: Path):
    # logging.info(f"ingestion_docx_path: {ingestion_docx_path}")
    # Create all folders if they don't exist
    ingestion_docx_path.mkdir(parents=True, exist_ok=True)  # parents=True for nested

    # 🧹 Clear previous files
    for file in ingestion_docx_path.iterdir():
        if file.is_file():
            file.unlink()  # delete file
        elif file.is_dir():
            # Optional: remove subdirectories too (recursive)
            import shutil
            shutil.rmtree(file)
    logging.info(f"🧹 Cleaned up folder: {ingestion_docx_path}")

    # get all docx in all nested folders
    file_globs = ["**/*.docx"]
    # Collect all docx path into docx_files
    docx_files = []
    for g in file_globs:
        docx_files.extend(sorted(kb_path.glob(g)))

    logging.info(f"Found {len(docx_files)} docx files to ingest under {kb_path}")

    for docx_fpath in docx_files:
        try:
            docx_path = Path(docx_fpath)  # get (Path) of docx # example: C:\Users\rag_minimal\WordDocument.docx
            # for debugging
            # docx_path = Path(r"C:\Users\xxxx\WordDocument.docx")

            # Get the stem (filename without suffix) and add " [rag]"
            new_stem = f"{docx_path.stem} [rag]"

            # Construct output file path inside the 'docx' folder
            output_path = ingestion_docx_path / f"{new_stem}.md" # C:\Users\rag_minimal\Ingestion\docx\WordDocument [rag].md
            # logging.info(f"output_path: {output_path}")

            # Convert docx to markdown
            docx2md.docx_to_markdown(docx_path=docx_fpath, output_path=output_path)

        except Exception as e:
            logging.info(f"⚠️ Skipping {docx_fpath} due to error: {e}")
            continue  # Explicitly continue to next file

# ----------------- Transform result (Query result to List) ---------------------------
def transform_result(result: dict) -> List[Dict[str, Any]]:
    top_records: List[Dict[str, Any]] = []

    id = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    print("Top matches: \n")
    for i, (doc, meta, d) in enumerate(zip(docs, metas, dists), start=1):
        sim = 1 - d
        source = meta.get("source_file") or meta.get("source") or meta.get("file") or "unknown"
        title = meta.get("title") or Path(source).stem
        record_type = meta.get("type") or meta.get("tag") or "unknown"
        identifier = meta.get("id") or meta.get("chunk_index") or title
        single_line = " ".join(doc.split())
        snippet = single_line[:140]
        if len(single_line) > 140:
            snippet = snippet.rstrip() + "..."

        # print(f"{i}. [{sim:.3f}] {title} | {record_type} | {identifier}")
        # print(f"    note: {title} | {snippet}")

        top_records.append({
            "rank": i,
            "score": float(d),
            "title": title,
            "type": record_type,
            "identifier": identifier,
            "snippet": snippet,
            "document": doc,
            "metadata": meta,
        })
    # print()
    return top_records


# ----------------- Query function (one shot example) -----------------
def chroma_query_with_rerank(query: str, collection) -> None:
    top_records: List[Dict[str, Any]] = []

    res = collection.query(
        query_texts=query,
        n_results=10,
        where={"type": "md"},
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    print("Top matches:")
    for i, (doc, meta, d) in enumerate(zip(docs, metas, dists), start=1):
        sim = 1 - d
        source = meta.get("source_file") or meta.get("source") or meta.get("file") or "unknown"
        title = meta.get("title") or Path(source).stem
        record_type = meta.get("type") or meta.get("tag") or "unknown"
        identifier = meta.get("id") or meta.get("chunk_index") or title
        single_line = " ".join(doc.split())
        snippet = single_line[:140]
        if len(single_line) > 140:
            snippet = snippet.rstrip() + "..."

        print(f"{i}. [{sim:.3f}] {title} | {record_type} | {identifier}")
        print(f"    note: {title} | {snippet}")

        top_records.append({
            "rank": i,
            "score": float(d),
            "title": title,
            "type": record_type,
            "identifier": identifier,
            "snippet": snippet,
            "document": doc,
            "metadata": meta,
        })
    print()

    # rerank
    rerank_pairs = rerank_with_cross_encoder(query, top_records) if top_records else []
    rerank_top_records: List[Dict[str, Any]] = []
    for rank, (rec, score) in enumerate(rerank_pairs, start=1):
        rerank_top_records.append(rec)
        print(f"{rank}. [{score:.3f}] {rec.get('title', '?')} | {rec.get('type', '?')}")
        print(f"    note: {rec.get('title')} | {rec.get('snippet')}")
    print()

    ans = maybe_answer_with_llm(query, rerank_top_records[:10]) if rerank_top_records else ""
    if ans:
        print("Answer (LLM):")
        print(ans)
        print()

# ----------------- Initialization helpers exported -----------------
def get_client(db_dir: Optional[str] = None):
    """Return a persistent chroma client (path points to DuckDB+parquet files)."""
    if db_dir is None:
        db_dir = str(Path(__file__).resolve().parent / "chroma_db")
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_dir))

def get_collection(
    name: str = "loans_kb",
    client: Optional[chromadb.PersistentClient] = None,
    db_dir: Optional[str] = None,
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    metadata: Optional[dict] = None,
):
    if client is None:
        client = get_client(db_dir=db_dir)
    if metadata is None:
        metadata = {"hnsw:space": "cosine"}
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata=metadata
    )
    return collection

# For convenience, do NOT create a global collection automatically; user decides when to create.
__all__ = [
    "get_client",
    "get_collection",
    "ingest_kb_to_collection",
]
