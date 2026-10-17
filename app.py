from pathlib import Path

import streamlit as st

from ingest import (
    build_vectorstore, clone_repo, get_embeddings, get_indexed_files,
    list_indexed_repos, load_files, load_vectorstore, split_documents,
)
from ragchain import answer_question, get_llm, project_overview

st.set_page_config(page_title="GitHub Repo Assistant", page_icon="💬", layout="wide")

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
    ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".c": "c",
    ".cpp": "cpp", ".h": "c", ".html": "html", ".css": "css", ".md": "markdown",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
}


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Loading language model (the first time is slow)...")
def load_llm():
    return get_llm()


for key, default in [("store", None), ("repo_name", None), ("files", []),
                     ("chunk_count", 0), ("messages", []), ("run_overview", False)]:
    st.session_state.setdefault(key, default)


def set_repo(name, store):
    st.session_state.store = store
    st.session_state.repo_name = name
    st.session_state.files = get_indexed_files(store)
    st.session_state.chunk_count = store._collection.count()
    st.session_state.messages = []


def index_repo(url, chunk_size):
    path = clone_repo(url)
    docs = load_files(path)
    chunks = split_documents(docs, chunk_size, chunk_size // 10)
    return path.name, build_vectorstore(chunks, path.name)


def build_tree(paths):
    tree = {}
    for p in paths:
        node = tree
        for part in p.split("/"):
            node = node.setdefault(part, {})
    lines = []

    def walk(node, prefix=""):
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0].lower()))  # folders first
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            lines.append(prefix + ("└── " if last else "├── ") + name + ("/" if child else ""))
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree)
    return "\n".join(lines)


def dedupe_sources(docs):
    sources, seen = [], set()
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            sources.append({
                "source": d.metadata["source"],
                "extension": d.metadata.get("extension", ""),
                "content": d.page_content,
            })
    return sources


def show_sources(sources):
    for src in sources:
        with st.expander(f"📄 {src['source']}"):
            st.code(src["content"], language=LANGUAGES.get(src["extension"]))


def respond(user_text, get_result):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            load_llm()
            with st.spinner("Searching the code and thinking..."):
                result = get_result()
            sources = dedupe_sources(result["sources"])
            st.markdown(result["answer"])
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": sources}
            )
        except RuntimeError as e:
            st.error(str(e))


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Repository")
    url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100,
                           help="Applies the next time you index a repo.")

    if st.button("Index repository", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Paste a GitHub URL first.")
        else:
            try:
                load_embeddings()
                with st.spinner("Cloning, splitting and embedding..."):
                    name, store = index_repo(url, chunk_size)
                set_repo(name, store)
                st.success(f"Indexed {name}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Indexing failed: {type(e).__name__}: {str(e)[:200]}")

    saved = list_indexed_repos()
    if saved:
        st.divider()
        choice = st.selectbox("Previously indexed repos", saved)
        if st.button("Load saved repo", use_container_width=True):
            try:
                load_embeddings()
                set_repo(choice, load_vectorstore(choice))
                st.success(f"Loaded {choice}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Could not load: {type(e).__name__}: {str(e)[:200]}")

    st.divider()
    k = st.slider("Chunks retrieved (k)", 1, 8, 5,
                  help="Raise this for questions that list things or span several files.")
    file_types = None

    if st.session_state.store is not None:
        files = st.session_state.files
        st.subheader(st.session_state.repo_name)
        c1, c2 = st.columns(2)
        c1.metric("Files", len(files))
        c2.metric("Chunks", st.session_state.chunk_count)

        options = sorted({Path(f).suffix.lower() for f in files if Path(f).suffix})
        selected = st.multiselect("Search only these file types", options, default=options,
                                  key=f"types_{st.session_state.repo_name}")
        if selected and len(selected) < len(options):
            file_types = selected

        with st.expander("Folder tree"):
            st.code(build_tree(files), language=None)

        if st.button("Project overview", use_container_width=True):
            st.session_state.run_overview = True

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- main area ----------------
st.title("GitHub Repo Assistant")
st.caption("Ask questions about a repository. Every answer shows the code it came from.")

if st.session_state.store is None:
    st.info("Paste a public GitHub URL and click **Index repository**, "
            "or load a repo you indexed before.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if st.session_state.run_overview:
    st.session_state.run_overview = False
    respond("Give me a project overview.", lambda: project_overview(st.session_state.store))

prompt = st.chat_input("Ask about the repository...", disabled=st.session_state.store is None)
if prompt:
    respond(prompt, lambda: answer_question(st.session_state.store, prompt, k=k, file_types=file_types))