# Local Memory-Based Contextual AI Assistant in Python

An intelligent, privacy-first personal AI assistant built in Python that understands user intent from current and previous conversations, retrieves relevant context from locally stored memory folders, and generates context-aware answers using free-tier AI APIs (such as Google Gemini) or offline mode.

---

## 🌟 Key Highlights & Architecture

- **Privacy-First & Local Storage**: All conversation transcripts and long-term memory notes reside entirely on your local machine in SQLite (`data/conversations.db`) and human-readable Markdown files (`memory/<category>/`). Only the minimal required context is sent to the AI API.
- **Lightweight & Fast**: Zero heavy local neural weight downloads (no gigabytes of PyTorch). Uses Google Gemini (`gemini-3.1-flash-lite` and `text-embedding-004`) for online AI tasks, coupled with an instant local n-gram TF-IDF and BM25 hybrid semantic fallback engine.
- **Pronoun & Coreference Resolution**: Resolves natural follow-up questions (e.g., *"How can I optimize a small language model for my laptop?"* followed by *"Can I make it run with less RAM?"*).
- **Categorized Local Memory**: Stores long-term notes in organized folders (`personal`, `college`, `family`, `career`, `exams`, `projects`, `preferences`, `technical_knowledge`, `general`) with YAML metadata headers.
- **Automatic Learning**: Detects genuine first-person disclosures and automatically classifies them into memories, while preventing false attributions from general or hypothetical questions.
- **Duplicate & Contradiction Handling**: Prevents duplicate memories and updates older facts when superseded.
- **Configurable Provider & Mock Mode**: Switch seamlessly between Google Gemini and an offline deterministic Mock AI provider for zero-cost testing.

---

## 📁 Directory Structure

```text
minor project/
├── main.py                     # Interactive Command-Line Interface REPL
├── config.py                   # Central configuration & path management
├── demo_followup_retrieval.py  # Demonstration: Follow-up across restarts
├── demo_category_retrieval.py  # Demonstration: Categorized memory targeting
├── requirements.txt            # Lightweight, lean dependencies
├── .env                        # Local environment settings
├── .env.example                # Example environment template
├── .gitignore                  # Git privacy rules (protects DB & memories)
│
├── core/
│   ├── assistant.py            # End-to-end Assistant orchestrator
│   ├── conversation_manager.py # SQLite conversation & message store
│   ├── similarity_engine.py    # Hybrid semantic & lexical similarity engine
│   ├── memory_manager.py       # Markdown memory store (CRUD + duplicates)
│   ├── memory_classifier.py    # Memory extraction & category classifier
│   ├── memory_retriever.py     # Multi-stage hybrid retrieval & ranking
│   ├── intent_detector.py      # Intent analysis & coreference resolution
│   ├── ai_client.py            # Google Gemini & Mock AI provider client
│   ├── efficiency/             # Algorithmic Non-AI compression & AST engine
│   │   ├── ast_compressor.py   # AST docstring & comment stripper
│   │   ├── text_compressor.py  # Pleonasm compactor & boilerplate pruning
│   │   ├── metrics.py          # Shannon entropy & multi-model cost savings
│   │   └── optimizer.py        # High-level prompt optimization coordinator
│   └── prompt_builder.py       # Context-aware prompt construction
│
├── memory/                     # Categorized local Markdown memories
├── data/
│   ├── conversations.db        # SQLite database with WAL mode
│   ├── memory_index/           # Persistent embedding & term cache
│   └── logs/                   # Application logs
│
└── tests/                      # Automated test suite (29 scenarios)
    ├── test_conversation.py
    ├── test_similarity.py
    ├── test_memory.py
    ├── test_retrieval.py
    ├── test_full_pipeline.py
    ├── test_modes.py           # 4-Mode validation & background memory tests
    └── test_efficiency.py      # AST compression, entropy & cost savings tests
├── templates/
│   ├── index.html              # Interactive Obsidian Memory Graph view
│   ├── chat.html               # Dedicated AI Chat interface with 4 modes
│   └── efficiency.html         # Token & Efficiency Optimizer workbench
└── gui_server.py               # Web application server (Chat, Graph & Optimizer)
```

---

## 🚀 Recommended Processing Pipeline

```text
User enters a question
        |
        v
Normalize and analyze the question
        |
        v
Identify intent and possible references (Pronoun/coreference resolution)
        |
        v
Retrieve recent conversation context (SQLite)
        |
        v
Search semantically similar past questions (Hybrid vector & lexical search)
        |
        v
Identify relevant memory categories
        |
        v
Retrieve and rank useful memories (Composite scoring: relevance, recency, confidence)
        |
        v
Check ambiguity and context conflicts
        |
        v
Build a compact contextual prompt (Strict fact/assumption separation)
        |
        v
Send the prompt to configured AI API (Google Gemini or Mock)
        |
        v
Receive and display the answer
        |
        v
Store the conversation locally in SQLite
        |
        v
Extract and update useful long-term memories in background
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Windows, macOS, or Linux

### 2. Clone or Navigate to Project
```bash
cd "d:/Work Hub/minor project"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
The assistant supports **FreeLLMAPI Unified Router** (OpenAI-compatible local proxy at `localhost:3001`), **Google Gemini API**, or **Mock mode** (offline).

To use your local FreeLLMAPI router, edit [`.env`](file:///d:/Work%20Hub/minor%20project/.env):
```ini
AI_PROVIDER=freellmapi
FREELLMAPI_KEY=your-freellmapi-key-here
FREELLMAPI_BASE_URL=http://localhost:3001/v1
FREELLMAPI_MODEL=gemini-3.1-flash-lite
EMBEDDING_PROVIDER=freellmapi
```

> **Note**: If no external API key is provided, the application automatically runs in `mock` mode, allowing complete offline use and testing.

---

## 💻 Running the Assistant

Start the interactive CLI:
```bash
python main.py
```

### Available Interactive Commands

Inside the assistant prompt, you can ask regular questions or use slash commands:

| Command | Description |
|---|---|
| `/help` | Display the interactive help menu |
| `/new [title]` | Start a new conversation session |
| `/history` | View previous conversation sessions |
| `/switch <id>` | Switch to an earlier conversation |
| `/view [id]` | View the complete message transcript |
| `/search <keyword>` | Search past message history |
| `/memories [category]` | List all stored long-term memories |
| `/view_memory <id>` | View complete Markdown content of a memory |
| `/add_memory` | Interactively add a new categorized memory |
| `/edit_memory <id>` | Edit an existing memory's content |
| `/delete_memory <id>` | Delete a memory by its ID |
| `/categories` | List active memory categories |
| `/add_category <name>`| Add a new memory category folder |
| `/provider <gemini\|mock>` | Switch active AI provider on the fly |
| `/stats` | View storage and message statistics |
| `/clear` | Clear the console screen |
| `/exit` | Exit the application safely |

---

## 🧪 Running Automated Tests

The automated test suite verifies all **14 required scenarios** using mock AI responses without consuming any API quota or requiring network access:

```bash
pytest -v
```

### Verified Scenarios:
1. `test_scenario_1_new_question_with_no_history`: New question with empty history.
2. `test_scenario_2_followup_question_referring_to_previous_message`: Pronoun resolution across turns.
3. `test_scenario_3_semantically_similar_question_different_wording`: Paraphrased question matching.
4. `test_scenario_4_similar_questions_different_intended_meanings`: Distinguishing opposite/divergent actions.
5. `test_scenario_5_retrieval_from_correct_category`: Targeting specific memory folders (`college`).
6. `test_scenario_6_retrieval_from_multiple_categories`: Bridging cross-category questions (`college` + `projects`).
7. `test_scenario_7_no_relevant_memory_found`: Responding without inventing personal facts.
8. `test_scenario_8_duplicate_memory_detection_and_updating`: Deduplication & confidence refresh.
9. `test_scenario_9_contradictory_memories_update`: Updating superseded facts.
10. `test_scenario_10_api_rate_limits_and_network_failures`: Graceful handling of HTTP 429 & network errors.
11. `test_scenario_11_empty_or_invalid_user_input`: Validation of blank/whitespace input.
12. `test_scenario_12_persistence_after_application_restart`: SQLite reloading across application restarts.
13. `test_scenario_13_memory_deletion_and_updates`: Deletion and index consistency.
14. `test_scenario_14_context_size_limits_for_long_conversations`: Sliding-window token budget management.

---

## 🌐 Web Interface: Chat, Memory Graph & Token Optimizer

The application features a modern web interface with three seamlessly connected sections:

1. **AI Chat (`/chat`)**:
   - 4 Independent Memory Modes:
     - **Normal Mode** (`memory_write: OFF`, `memory_retrieval: OFF`): Ultra-fast direct chat with current conversation context only. No memory lookups or extractions.
     - **Memory Mode** (`memory_write: ON`, `memory_retrieval: OFF`): Automatically detects and extracts new facts into your categorized memory notes in the background.
     - **Retrieval Mode** (`memory_write: OFF`, `memory_retrieval: ON`): Recalls existing memories to provide personalized answers without writing new memories.
     - **Both (Combined)** (`memory_write: ON`, `memory_retrieval: ON`): Full assistant that both retrieves verified context and updates knowledge.
   - Live model selector (`gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `auto`, etc.) with resilient 429 fallback.
   - Conversation history sidebar with search, title renaming, and deletion.
   - Markdown rendering, code syntax highlighting, copy-to-clipboard buttons, and timestamps.

2. **Memory Graph (`/memory-graph` or `/`)**:
   - Obsidian-style force-directed interactive D3.js visualization.
   - Category clustering, tags, semantic links, and full memory CRUD management.

3. **Token & Efficiency Optimizer (`/efficiency`)**:
   - Client-side algorithmic pre-processing and Non-AI token compression.
   - **AST Code Compression**: Strips docstrings and comments from Python, JS, SQL, and C++ with mathematical AST equivalence verification.
   - **Rule-Based Text Compaction**: Prunes pleonasms and conversational fluff while strictly preserving Semantic Anchors (`not`, `must`, `o(1)`).
   - **Information Density**: Computes Shannon Entropy $H(X)$ (bits/token) and lexical density gains.
   - **Financial Cost Calculator**: Multi-model savings estimation (Gemini, GPT-4o, Claude 3.5, and Edge SLMs) per query and for 10,000 queries.
   - **One-Click Send to Chat**: Send optimized prompts straight into the active AI Chat session.

### Starting the Web Application:
```bash
python gui_server.py
```
- **AI Chat**: [http://localhost:5050/chat](http://localhost:5050/chat)
- **Memory Graph**: [http://localhost:5050/memory-graph](http://localhost:5050/memory-graph)
- **Token Optimizer**: [http://localhost:5050/efficiency](http://localhost:5050/efficiency)

---

## 🎬 Running the Demonstrations

### Demonstration 1: Follow-Up Retrieval Across Restarts
Demonstrates how the assistant conducts a conversation in Session 1, completely shuts down, restarts in Session 2, and answers a follow-up referring to "it" without the user re-explaining the context:
```bash
python demo_followup_retrieval.py
```

### Demonstration 2: Categorized Memory Retrieval & Category Targeting
Demonstrates categorized Markdown notes, automatic category identification, and answering questions from specific memory categories:
```bash
python demo_category_retrieval.py
```

---

## 🔒 Privacy & Data Protection

- **Local Storage by Default**: Conversations are recorded exclusively in `data/conversations.db` on your local disk.
- **Categorized Memory Privacy**: Personal memories are saved locally in `memory/<category>/*.md`.
- **Minimal Context Transmission**: When calling the external AI model, only the active turn and filtered relevant memory excerpts are sent; the entire memory folder is never uploaded.
- **Git Exclusions**: `.env`, private databases (`data/*.db`), and user memory markdown files are excluded in `.gitignore`.
- **Sensitive Information Safeguard**: Passwords, API tokens, and confidential personal data are detected and prevented from being stored into automatic memories.

---

## ⚙️ Configuration Reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `"freellmapi"` | Active provider: `freellmapi`, `gemini`, `mock`, or `auto` |
| `FREELLMAPI_KEY` | `""` | FreeLLMAPI Unified Router API Key |
| `FREELLMAPI_BASE_URL` | `"http://localhost:3001/v1"` | FreeLLMAPI Proxy Base URL |
| `FREELLMAPI_MODEL` | `"auto"` | Model name routed by FreeLLMAPI |
| `EMBEDDING_PROVIDER` | `"freellmapi"` | `freellmapi`, `gemini`, or `local` (TF-IDF fallback) |
| `GEMINI_API_KEY` | `""` | Google Gemini API Key (optional fallback) |
| `GEMINI_MODEL` | `"gemini-3.1-flash-lite"` | Gemini direct model |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity for past question matches |
| `MEMORY_RELEVANCE_THRESHOLD` | `0.30` | Minimum composite score for memory retrieval |
| `MAX_RECENT_MESSAGES` | `8` | Maximum messages in recent context window |
| `MAX_RETRIEVED_MEMORIES` | `5` | Maximum long-term memory items passed to prompt |
| `MAX_CONTEXT_TOKENS` | `3000` | Token budget limit for prompt assembly |
| `MEMORY_AUTO_STORE_CONFIDENCE` | `0.65` | Minimum confidence score to store auto-memory |
