import os
import shutil
import stat
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path

from git import Repo
from git.exc import GitCommandError
from langchain_core.documents import Document
import re
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

REPOS_DIR = Path("repos")

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".html", ".css",
    ".md", ".json", ".yaml", ".yml", ".txt",
}
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "build", "dist", ".idea", ".vscode",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock",
}
MAX_FILE_SIZE = 200 * 1024  # 200 KB


def _remove_readonly(func, path, _exc):
    # Windows marks .git files read-only, which breaks shutil.rmtree
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(url: str) -> Path:
    """Shallow-clone a public GitHub repo into repos/<name> and return the path."""
    url = url.strip().rstrip("/")
    if not url.startswith("https://github.com/"):
        raise ValueError("Please enter a GitHub URL like https://github.com/owner/repo")

    name = url.removesuffix(".git").split("/")[-1]
    if not name:
        raise ValueError("Could not read the repository name from that URL.")

    REPOS_DIR.mkdir(exist_ok=True)
    dest = REPOS_DIR / name
    if dest.exists():
        shutil.rmtree(dest, onerror=_remove_readonly)

    try:
        # GIT_TERMINAL_PROMPT=0 makes private repos fail fast instead of asking for a password
        Repo.clone_from(url, dest, depth=1, env={"GIT_TERMINAL_PROMPT": "0"})
    except GitCommandError:
        raise ValueError("Could not clone. The repo may be private or the URL is wrong.")
    return dest


def load_files(path) -> list[Document]:
    """Walk a folder and return one Document per useful text file."""
    root = Path(path)
    docs = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]  # prune junk folders

        for filename in filenames:
            file_path = Path(dirpath) / filename
            ext = file_path.suffix.lower()

            if filename in SKIP_FILES or ext not in ALLOWED_EXTENSIONS:
                continue
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip() or "\x00" in text:  # skip empty or binary files
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.relative_to(root).as_posix(),
                        "extension": ext,
                    },
                )
            )

    if not docs:
        raise ValueError("No readable code or text files were found in this repository.")
    return docs
EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".h": Language.C,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
}


def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for doc in docs:
        language = EXTENSION_TO_LANGUAGE.get(doc.metadata["extension"])
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        chunks.extend(splitter.split_documents([doc]))
    return [c for c in chunks if len(c.page_content.strip()) >= 50]   # <- 4 spaces, same level as "for"

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _collection_name(name: str) -> str:
    # Chroma needs 3-63 chars, starting and ending with a letter or digit
    clean = re.sub(r"[^a-zA-Z0-9_-]", "-", name).strip("-_")
    return (clean or "repo").ljust(3, "x")[:63]


def build_vectorstore(chunks, name):
    """Embed chunks and save them in a fresh Chroma collection for this repo."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _collection_name(name)

    try:
        client.delete_collection(collection)  # avoid duplicates when re-indexing
    except Exception:
        pass  # collection didn't exist yet

    store = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=get_embeddings(),
    )
    for i in range(0, len(chunks), 500):  # batches keep memory use steady
        store.add_documents(chunks[i : i + 500])
    return store


def load_vectorstore(name):
    """Reopen a previously indexed repo without re-embedding."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _collection_name(name)
    try:
        client.get_collection(collection)
    except Exception:
        raise ValueError(f"No saved index found for '{name}'. Index the repo first.")
    return Chroma(
        client=client,
        collection_name=collection,
        embedding_function=get_embeddings(),
    )
def list_indexed_repos():
    """Names of repos already saved in chroma_db/."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    return sorted(names)


def get_indexed_files(store):
    """Sorted list of source file paths stored in a collection."""
    data = store.get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})

if __name__ == "__main__":
    repo_path = clone_repo("https://github.com/karpathy/nanoGPT")
    documents = load_files(repo_path)
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"Embedding {len(chunks)} chunks...")

    store = build_vectorstore(chunks, "nanoGPT")
    print("Stored. Collection size:", store._collection.count())