import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()  # "groq" or "local"

# Groq: the primary model is tried first, then each backup in order.
LLM_MODEL_GROQ = os.getenv("LLM_MODEL_GROQ", "openai/gpt-oss-120b")
LLM_MODEL_GROQ_BACKUPS = [
    m.strip()
    for m in os.getenv("LLM_MODEL_GROQ_BACKUPS", "qwen/qwen3.8-27b,openai/gpt-oss-20b").split(",")
    if m.strip()
]

# Local: free, runs on your own machine (slow on CPU)
LLM_MODEL_LOCAL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

NOT_FOUND = "I couldn't find that in the repository."

ENTRY_NAMES = (
    "app.py", "main.py", "chat.py", "manage.py",
    "index.js", "server.js", "index.ts",
)

SYSTEM_PROMPT = f"""You are a code assistant that answers questions about a GitHub repository.
Answer ONLY using the code and text in the CONTEXT below.
Mention the file names your answer is based on.
Keep the answer under 100 words unless code is needed.
If the answer is not in the context, reply exactly: {NOT_FOUND}
Do not guess or invent files, functions, or behavior."""

OVERVIEW_PROMPT = """You are a code assistant. Using ONLY the CONTEXT below
(the README and main files of a GitHub repository), write a short
project overview: what the project does, the main files and what
each is for, and how to run it. Use at most 150 words.
Do not invent anything that is not in the context."""


@lru_cache(maxsize=1)
def get_llm():
    """Create the selected chat model once and reuse it."""
    if LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file or the host's secrets."
            )

        def make(model_name):
            return ChatGroq(
                model=model_name,
                temperature=0,
                max_tokens=1024,
                timeout=60,
                max_retries=1,
            )

        primary = make(LLM_MODEL_GROQ)
        backups = [make(m) for m in LLM_MODEL_GROQ_BACKUPS]
        return primary.with_fallbacks(backups) if backups else primary

    if LLM_BACKEND == "local":
        import torch
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        torch.set_num_threads(os.cpu_count() or 4)
        llm = HuggingFacePipeline.from_model_id(
            model_id=LLM_MODEL_LOCAL,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 250,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=llm)

    raise RuntimeError(f"Unknown LLM_BACKEND '{LLM_BACKEND}'. Choose 'groq' or 'local'.")


def _friendly_error(e):
    """Short message for the page. Full detail goes to the terminal."""
    if isinstance(e, RuntimeError):  # our own clear messages
        return str(e)

    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    text = str(e).lower()
    print(f"[LLM error] {type(e).__name__}: {str(e)[:300]}")

    if status in (401, 403) or "invalid_api_key" in text:
        return "The demo's API key was rejected. The app owner needs to check it."
    if status == 429 or "rate_limit" in text:
        return "The demo is busy right now. Please wait a minute and try again."
    if status == 404 or "model_not_found" in text:
        return "The language model is unavailable right now. Please try again later."
    return "The language model call failed. Please try again in a moment."


def format_context(docs):
    return "\n\n---\n\n".join(
        f"File: {d.metadata['source']}\n{d.page_content}" for d in docs
    )


def _clean_answer(text):
    """Remove <think>...</think> blocks that some models print, and tidy whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.strip()


def _ask(system_prompt, docs, question):
    if not docs:  # nothing retrieved: don't let the model guess
        return {"answer": NOT_FOUND, "sources": []}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"CONTEXT:\n{format_context(docs)}\n\nQUESTION: {question}"),
    ]

    try:
        response = get_llm().invoke(messages)
    except Exception as e:
        raise RuntimeError(_friendly_error(e)) from e

    answer = _clean_answer(response.content)
    if not answer:
        answer = "The model returned an empty answer. Please try again or rephrase the question."
    return {"answer": answer, "sources": docs}


def retrieve(store, question, k=5, file_types=None):
    kwargs = {"filter": {"extension": {"$in": list(file_types)}}} if file_types else {}
    return store.similarity_search(question, k=k, **kwargs)


def answer_question(store, question, k=5, file_types=None):
    """Return {'answer': str, 'sources': [Document, ...]}."""
    question = question.strip()
    if not question:
        return {"answer": "Please type a question about the repository.", "sources": []}
    return _ask(SYSTEM_PROMPT, retrieve(store, question, k, file_types), question)


def project_overview(store):
    """Summarize the repository from its README and entry-point files."""
    data = store.get(include=["metadatas"])
    sources = sorted({m["source"] for m in data["metadatas"]})

    readme = next((s for s in sources if s.lower().startswith("readme")), None)
    entry_files = [
        s for s in sources if s.rsplit("/", 1)[-1].lower() in ENTRY_NAMES
    ][:3]

    docs = []
    for name, limit in [(readme, 3)] + [(f, 1) for f in entry_files]:
        if not name:
            continue
        got = store.get(where={"source": name}, include=["documents", "metadatas"])
        docs += [
            Document(page_content=d, metadata=m)
            for d, m in zip(got["documents"], got["metadatas"])
        ][:limit]

    if not docs:  # fallback for repos with no README or known entry files
        docs = store.similarity_search("main entry point that runs the application", k=4)

    return _ask(OVERVIEW_PROMPT, docs, "Give a short overview of this project.")


if __name__ == "__main__":
    from ingest import load_vectorstore

    store = load_vectorstore("nanoGPT")
    for q in [
        "What does the configurator.py file do?",
        "Where is the database connection handled?",
    ]:
        print("\nQ:", q, flush=True)
        print("A:", answer_question(store, q)["answer"])

    print("\nOVERVIEW:", project_overview(store)["answer"])