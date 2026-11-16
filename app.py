"""PDF Chat: a fully local RAG app (Streamlit + Ollama). No API keys needed."""
import json

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's PDF documents. "
    "Use ONLY the context provided. If the answer is not in the context, say you could "
    "not find it in the documents. Be concise and cite sources like [file.pdf p.3]."
)


# ---------- Ollama helpers ----------
def embed(texts):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype="float32")


def stream_chat(model, messages):
    with requests.post(
        f"{OLLAMA}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "message" in data:
                yield data["message"].get("content", "")
            if data.get("done"):
                break


def ollama_ok():
    try:
        requests.get(OLLAMA, timeout=2)
        return True
    except requests.RequestException:
        return False


# ---------- RAG pipeline ----------
def chunk_pdf(file, size=800, overlap=150):
    """Split each page into overlapping chunks, remembering file and page number."""
    reader = PdfReader(file)
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append({"text": piece, "page": page_no, "file": file.name})
            start += size - overlap
    return chunks


def index_files(files):
    """Embed chunks of new PDFs and add them to the in-memory index."""
    for f in files:
        key = f"{f.name}:{f.size}"
        if key in st.session_state.indexed:
            continue
        chunks = chunk_pdf(f)
        if not chunks:
            st.sidebar.warning(f"No text found in {f.name} (scanned PDF?).")
            continue
        bar = st.sidebar.progress(0.0, text=f"Indexing {f.name}…")
        vecs = []
        for i in range(0, len(chunks), 32):
            batch = [c["text"] for c in chunks[i:i + 32]]
            vecs.append(embed(batch))
            bar.progress(min((i + 32) / len(chunks), 1.0))
        bar.empty()
        v = np.vstack(vecs)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        st.session_state.chunks += chunks
        st.session_state.vectors = (
            v if st.session_state.vectors is None
            else np.vstack([st.session_state.vectors, v])
        )
        st.session_state.indexed.add(key)


def retrieve(query, k):
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-9
    scores = st.session_state.vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [(st.session_state.chunks[i], float(scores[i])) for i in top]


# ---------- UI ----------
st.set_page_config(page_title="PDF Chat (Local RAG)", page_icon="📄")
for key, default in [("chunks", []), ("vectors", None), ("indexed", set()), ("history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📄 Chat with your PDFs")
st.caption("Runs 100% locally with Ollama. Your documents never leave your computer.")

with st.sidebar:
    st.header("Settings")
    chat_model = st.text_input("Chat model", "llama3.2")
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Clear everything"):
        st.session_state.clear()
        st.rerun()

if not ollama_ok():
    st.error("Ollama is not running. Start it, then run: ollama pull llama3.2 and ollama pull nomic-embed-text")
    st.stop()

if files:
    try:
        index_files(files)
    except requests.RequestException as e:
        st.sidebar.error(f"Embedding failed. Did you run 'ollama pull {EMBED_MODEL}'? ({e})")

if st.session_state.vectors is not None:
    st.sidebar.success(f"{len(st.session_state.indexed)} PDF(s), {len(st.session_state.chunks)} chunks indexed")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(f"{s['file']} · page {s['page']} · score {s['score']:.2f}")
                    st.write(s["text"])

question = st.chat_input("Ask something about your documents…")
if question:
    if st.session_state.vectors is None:
        st.warning("Upload at least one PDF first.")
        st.stop()
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question, top_k)
    context = "\n\n".join(f"[{c['file']} p.{c['page']}] {c['text']}" for c, _ in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state.history[-6:]]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_chat(chat_model, messages))
        except requests.RequestException as e:
            answer = f"Model error. Did you run 'ollama pull {chat_model}'? ({e})"
            st.error(answer)
        sources = [{**c, "score": s} for c, s in hits]
        with st.expander("Sources"):
            for s in sources:
                st.caption(f"{s['file']} · page {s['page']} · score {s['score']:.2f}")
                st.write(s["text"])

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
