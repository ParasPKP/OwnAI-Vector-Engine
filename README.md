# 🗄️ OwnAI Vector Engine — Vector Database from Scratch in Java

A fully working **Vector Database** built from scratch in Java with a web UI powered by OwnAI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side,
plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to understand how production vector databases
> like Pinecone, Weaviate, and Chroma actually work under the hood. Fully open-source and self-hosted on your machine.

---

## ✨ Features

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |
| **Zero Dependencies** | Pure Java — no Maven, no Gradle, no external libraries |

---

## 🧠 How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)       ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Java)               ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                 ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)               ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone,
Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is
progressively sparser — searches start at the top layer and zoom in, achieving
O(log N) complexity instead of O(N) for brute force.

---

## 🛠️ Prerequisites

You need **3 things** installed:

1. **Java JDK 17+**
2. **IntelliJ IDEA** (free Community edition works fine)
3. **Ollama** — for the AI/embedding features *(optional for basic search)*

---

## 🚀 Setup — Step by Step

### Step 1 — Install Java JDK

1. Go to **https://adoptium.net** and download **Java 17** (Temurin)
2. Run the installer — make sure **"Add to PATH"** is checked
3. Verify in Command Prompt:
   ```
   java -version
   ```
   You should see something like `openjdk version "17.x.x"`

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/ParasPKP/OwnAI-Vector-Engine.git
cd OwnAI-Vector-Engine
```

Or download the ZIP from GitHub → click the green **Code** button → **Download ZIP** → extract it.

---

### Step 3 — Open in IntelliJ IDEA

1. Open IntelliJ IDEA
2. Click **Open** → select the `OwnAI-Vector-Engine` folder
3. IntelliJ will detect it as a Java project automatically
4. Wait for it to finish indexing (progress bar at the bottom)

---

### Step 4 — Run the Server

1. Open `src/Main.java` in the editor
2. Find the `main` method — you'll see a green **▶ play button** in the left margin
3. Click it → **Run 'Main.main()'**
4. The Run panel at the bottom will show:

```
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: OFFLINE (install from https://ollama.com)
Server started → http://localhost:8080
```

5. Open your browser and go to **http://localhost:8080** 🎉

---

### Step 5 — Install Ollama (for AI features)

1. Go to **https://ollama.com** → Download for Windows → run the installer
2. Open **Command Prompt** and pull the two required models:

```bash
ollama pull nomic-embed-text
```
*(~274 MB — the embedding model)*

```bash
ollama pull llama3.2
```
*(~2 GB — the language model)*

3. Re-run the project in IntelliJ — it should now say `Ollama: ONLINE`

> **Minimum specs for Ollama:** 8 GB RAM recommended.

---

## 🖥️ Using the Application

### Tab 1 — Search (Demo Vectors)

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose an algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose a distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

The scatter plot shows all 20 vectors projected to 2D using PCA. Notice how the
4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is
what "semantic similarity" looks like visually.

### Tab 2 — Documents (Real Embeddings)

Requires Ollama to be running.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, Wikipedia articles, textbook paragraphs
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own 768D embedding stored in a separate HNSW index

### Tab 3 — Ask AI (RAG Pipeline)

Requires Ollama and at least one document inserted in Tab 2.

1. Type a question about your documents
2. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question  →  embedded with nomic-embed-text (768D vector)
2. HNSW search    →  finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2       →  generates an answer based on your documents
```

---

## 📡 REST API Reference

The server runs at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a new demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## 📁 Project Structure

```
OwnAI-Vector-Engine/
├── src/
│   └── Main.java      ← entire backend: HNSW, KD-Tree, BruteForce, REST API, RAG
├── index.html         ← frontend: PCA scatter plot, chat UI, benchmark
├── .gitignore
└── README.md
```

### Architecture (Main.java)

```
BruteForce     O(N·d)     Exact, baseline — checks every single vector
KDTree         O(log N)   Exact, splits space along axes to prune search
HNSW           O(log N)   Approximate, multilayer small-world graph

VectorDB       Unified interface over all 3 (16D demo vectors)
DocumentDB     HNSW-only index for real Ollama embeddings (768D)
OllamaClient   HTTP client → /api/embeddings + /api/generate
```

---

## 📖 Algorithm Explanations

### HNSW (Hierarchical Navigable Small World)

Think of it like a road network. Layer 0 is every small street (all nodes, many connections).
Layer 1 is major roads (fewer nodes). Layer 2 is highways (very few nodes, long-range jumps).

When **inserting**, a node is randomly assigned a max layer. The algorithm descends
from the top layer, finding nearest neighbors and connecting bidirectionally at each layer.

When **searching**, it enters at the top (fast highway navigation), zooms to the right
neighborhood, then zooms in at layer 0 for precise results.

This is why it's **O(log N)** — upper layers skip huge portions of the graph.

### KD-Tree (K-Dimensional Tree)

Recursively splits space in half along one dimension at a time (like a binary search tree,
but in N dimensions). During search, entire subtrees are skipped if they can't possibly
contain a closer point than what we've already found.

**Weakness:** Works great up to ~20 dimensions. At 768D (Ollama embeddings), almost
nothing gets pruned and it degrades to brute force.

### Brute Force

Calculates distance to every single vector. Always exact, always slowest.
Useful as a correctness baseline when comparing algorithms.

---

## ❗ Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in the header | Run `ollama serve` in Command Prompt |
| Embedding takes a long time | Ollama is downloading the model on first use — wait 2 min |
| Port 8080 already in use | Change `PORT = 8080` in `Main.java` to `8081` and rerun |
| `index.html not found` error | Make sure `index.html` is in the project root, not inside `src/` |
| LLM answer is very slow | Switch to the faster 1B model (see below) |

### Use a Smaller / Faster LLM

If `llama3.2` is too slow on your laptop, use the 1B model instead:

```bash
ollama pull llama3.2:1b
```

Then in `Main.java` find this line and change it:

```java
String genModel = "llama3.2";      // change to:
String genModel = "llama3.2:1b";
```

Re-run the project in IntelliJ.

---

## 🤝 Contributing

This is a learning project — contributions, improvements, and questions are all welcome!

1. Fork the repository
2. Create a branch: `git checkout -b my-feature`
3. Commit your changes: `git commit -m "add my feature"`
4. Push: `git push origin my-feature`
5. Open a **Pull Request**

---

## 👨‍💻 Developer

- Repository: https://github.com/ParasPKP/OwnAI-Vector-Engine.git
- GitHub: https://github.com/ParasPKP
- LinkedIn: https://www.linkedin.com/in/paras-parshuramkar-b8237b315/

---

## 📄 License

MIT License
