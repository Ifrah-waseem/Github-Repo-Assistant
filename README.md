# GitHub Repo Assistant

Chat with any public GitHub repository. Paste a URL, the app clones and indexes the code, and then answers questions about it. Every answer shows the files and code chunks it came from, so you can check it yourself.

![Screenshot](docs/screenshot-chat.png)
<img width="309" height="400" alt="p1" src="https://github.com/user-attachments/assets/47593f53-8beb-4d69-a66e-97df38b3d3f5" />
<img width="312" height="450" alt="p2" src="https://github.com/user-attachments/assets/1d89e87a-11c8-4f29-8f91-45b09a75ef2a" />
<img width="659" height="387" alt="p3" src="https://github.com/user-attachments/assets/8e215004-3200-4e30-982e-c5fece626aaa" />

##APP LINK:https://app-repo-assistant-acqq47hpfzmya3s8y2uy4p.streamlit.app/
## Features

- Index any public GitHub repository from its URL (shallow clone, default branch)
- Language-aware code splitting that tries to keep functions and classes together
- Semantic search over the code with a local embedding model and a persistent ChromaDB store
- Answers grounded in the retrieved code, with file names and source expanders
- Refuses to answer ("I couldn't find that in the repository.") when the context does not contain the answer
- Project overview button (summary from the README and main files)
- Folder tree, file-type filter, and adjustable chunk size and number of retrieved chunks (k)
- Reload previously indexed repositories without re-embedding
- Several LLM backends: Groq (with automatic fallback models) or a free local Hugging Face model

## How it works

```
GitHub URL
   |
   v
1. Clone         shallow clone with GitPython
2. Load          read useful code and text files, skip junk and large files
3. Split         language-aware chunks (about 1000 characters, 10% overlap)
4. Embed         BAAI/bge-small-en-v1.5 (sentence-transformers)
5. Store         ChromaDB, one collection per repository, saved to disk
   |
   v  (question)
6. Retrieve      top-k similar chunks, optionally filtered by file type
7. Generate      LLM answers using only those chunks
8. Display       Streamlit chat with an expander per source chunk
```

## Tech stack

Python, LangChain, ChromaDB, Hugging Face (embeddings, optional local LLM), Groq (hosted LLM), Streamlit, GitPython.

## Project structure

```
app.py            Streamlit interface
ingest.py         clone, load, split, embed, store, reload
ragchain.py       retrieval, prompts, LLM backends, answers
requirements.txt  Python dependencies
packages.txt      system packages for hosting (git)
.gitignore        keeps .env, venv, repos/ and chroma_db/ out of Git
```

## Setup (local)

Requires Python 3.10 or newer and Git installed on your machine.

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder (one variable per line, never commit it):

```
GROQ_API_KEY=your_groq_key
LLM_BACKEND=groq
```

Run the app:

```bash
streamlit run app.py
```

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (needed for the `groq` backend) | none |
| `LLM_BACKEND` | `groq` or `local` | `groq` |
| `LLM_MODEL_GROQ` | Primary Groq model | `openai/gpt-oss-120b` |
| `LLM_MODEL_GROQ_BACKUPS` | Comma-separated fallback models, tried in order | `qwen/qwen3.8-27b,openai/gpt-oss-20b` |
| `HF_TOKEN` | Optional. Removes the Hugging Face download warning | none |

Groq model availability and limits differ by account and change over time. Check your own limits in the Groq console and set the model variables to models your account can use.

### Free local mode

Set `LLM_BACKEND=local` to run `Qwen/Qwen2.5-Coder-1.5B-Instruct` on your own machine. It needs no API key and no payment, but it downloads about 3 GB on first use, needs roughly 6 GB of RAM, and is slow on CPU (tens of seconds per answer). It is also less reliable at following the "answer only from the context" rule.

## Deployment notes

- Store `GROQ_API_KEY` and `LLM_BACKEND=groq` as **secrets** on the host, never in the code.
- `packages.txt` contains `git`, which the clone step needs. Check your host's documentation for how it installs system packages.
- The embedding model runs on the host and uses PyTorch, so choose a host with enough memory. Check its current limits first.
- Many free hosts use temporary disk. Indexed repositories can disappear when the app restarts, so visitors may need to index a repo again.


## Limitations

This is a learning and portfolio project. These are the known limits, grouped by where they come from.

### Retrieval quality

- **Prose can outrank code.** For broad "how does X work?" questions, README text and HTML can fill the top results while the code that implements the behavior is missed. In testing, "How does the frontend start the interview?" retrieved the HTML form and README but not the JavaScript handler. A code-flavored question ("What does script.js do when the start form is submitted?") found it.
- **List-style questions can be wrongly refused.** Questions like "Which endpoints does the backend expose?" depend on lines spread across several chunks. If one is missed, the app says it could not find the answer even though it exists. Raising k or filtering to the right file type fixed it in testing.
- **Semantic search is weak on exact names.** Searching for an exact function, variable or constant name can fail, because there is no keyword (BM25) search or re-ranking.
- **Long functions are split.** A function longer than the chunk size is cut, so a chunk may start mid-function without its `def` line, and the model cannot see which function it belongs to.
- **Cross-file questions are hard.** Questions that need several files at once ("how do these two parts interact?") only work if all the right chunks land in the top k.
- **No conversation memory.** Each question is answered independently, so follow-ups like "what about the second file?" do not work.
- **Very small chunks are dropped.** Chunks under 50 characters are removed, which can discard tiny files or one-line snippets.

### Answer quality

- **The model can still be wrong.** Grounding in retrieved code reduces invented answers, but the model can misread the code or confuse similar functions. In testing, the small local model mixed up two similar form handlers.
- **Partial answers.** When only part of the relevant code is retrieved, the answer can be true but incomplete.
- **Reasoning models can return empty or cut-off text.** The app replaces empty replies with a retry message, but answers may still be shorter than expected.
- **Answers reflect only what was retrieved.** The model never sees the whole repository.
- **Local mode is weaker and slower.** The 1.5B model follows instructions less reliably than the hosted models.

### What gets indexed

- **Public GitHub repositories only.** The URL must start with `https://github.com/`. Private repositories, other hosts and non-repository links are rejected.
- **Default branch, latest commit only.** The clone is shallow, so there is no history, no other branches and no tags.
- **A fixed set of file types.** Currently `.py .js .ts .jsx .tsx .java .go .rs .c .cpp .h .html .css .md .json .yaml .yml .txt`. Other types are ignored, including Arduino `.ino`, Jupyter `.ipynb`, `.php`, `.cs`, `.kt`, `.swift` and `.rb`. A repository made only of unsupported files cannot be indexed.
- **Large and generated files are skipped.** Files over about 200 KB, lock files, images, binaries, and folders such as `node_modules`, `venv`, `build` and `dist` are skipped, so some content is never searchable.
- **No incremental updates.** If the repository changes, you must re-index it. Re-indexing replaces the old index.
- **Large repositories are slow.** Embedding runs on the machine hosting the app, usually on CPU.
- **Two repositories with the same name collide.** The index and clone folder are named after the repository only (not the owner), so `owner-a/app` and `owner-b/app` overwrite each other.

### Hosting and scale

- **Shared collections.** Indexes are stored on the server and keyed by repository name. If several visitors index the same repository at once, they can replace each other's index mid-session.
- **Saved repositories may not persist.** On hosts with temporary storage, the "previously indexed repos" list can be empty after a restart.
- **Memory limits.** The embedding model needs PyTorch. Small free hosts may run out of memory or start slowly.
- **First request is slow.** The embedding model downloads on first use.
- **System dependencies.** Git must be installed on the host. ChromaDB can also fail on hosts with an old SQLite version.
- **No user accounts, authentication or usage limits.**

### API limits and cost

- **Free-tier rate limits are shared by every visitor.** Groq limits requests and tokens per minute and per day, per model, for the whole account. Heavy use returns a "demo is busy" message. The fallback models help, but all of them can run out.
- **Limits and model availability change.** A model that works today may be removed or restricted. Update the model names in the settings if that happens.
- **A larger k uses more tokens.** Each question sends several code chunks, so a higher k means fewer questions fit in the rate limit.

### Privacy and security

- **Code is sent to a third party.** Retrieved chunks go to the LLM provider. Free tiers usually give no privacy guarantee, so use public repositories only.
- **Secrets in repositories can be shown.** The source expanders display raw code. If a repository contains hard-coded passwords or API keys, the app can display them.
- **Prompt injection.** Text inside a repository (a README or a comment) can contain instructions aimed at the model. The system prompt tells it to answer only from the context, but this is not a hard guarantee.
- **Open to anyone with the link.** Visitors spend the owner's API quota.
- 
### Interface

- The chunk size slider only takes effect when a repository is indexed again.
- Answers appear all at once (no streaming).
- The source list shows retrieved chunks, not a precise line number or a guarantee that a chunk was actually used by the model.

## Possible improvements

- Hybrid search (BM25 plus vectors) for exact names and list-style questions
- A cross-encoder re-ranker
- Conversation memory with a history-aware retriever
- Streaming answers
- Support for more file types (`.ino`, `.ipynb`, and others)
- Collection names that include the repository owner, and per-session indexes
- A reproducible evaluation script with a larger, more varied test set
- An agent that decides whether to search, open a specific file, or answer directly
