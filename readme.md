# Offline Science AI Chatbot - RAG + LLM  
## AIML Internship Assessment Submission

| Category           | Detail                                     |
|--------------------|--------------------------------------------|
| **Author**         | Siddharth Badjate                          |
| **Domain**         | Science (Physics, Chemistry, Biology)     |
| **Dataset**        | SciQ (13,679 Question & Answer pairs)     |
| **Retrieval Model**| Sentence-Transformers (all-MiniLM-L6-v2)  |
| **Generation Model**| Qwen-2.5-0.5B-Instruct                     |
| **Framework**      | Retrieval-Augmented Generation (RAG)      |

---

## 1. Overview and Core Functionality

This project delivers a fully offline, self-contained AI chatbot designed to serve as a science tutor. It utilizes the power of Retrieval-Augmented Generation (RAG) to provide accurate, contextually grounded answers to specific science questions, overcoming the inherent limitations of small, general-purpose Large Language Models (LLMs) by augmenting them with a specialized knowledge base.

### Key Features
- **Offline Operation:** Once initial setup (downloading models and data) is complete, the application requires no internet connection to function, making it ideal for restricted or low-connectivity environments.  
- **Semantic RAG Architecture:** The system dynamically fetches knowledge from the *SciQ* dataset using semantic search, ensuring generated answers are factually correct and sourced.  
- **Intelligent Query Pre-processing:** Includes automatic spelling correction (pyspellchecker) on the user query (e.g., `"mitrochondira" → "mitochondria"`), significantly improving retrieval accuracy against the knowledge base.  
- **User-Friendly GUI:** A responsive, Tkinter-based chat interface provides an intuitive user experience with timestamps, distinct user/bot styling, and source citation.  
- **Robust Error Handling:** Implements a graceful fallback mechanism that shifts to a retrieval-only summary mode if the resource-intensive LLM (Qwen-0.5B) fails to load or execute.

---

## 2. Detailed RAG Architecture

The RAG pipeline is implemented in the `RAGWithLLM` class, following a three-stage process: Data Preparation, Retrieval, and Generation.

### 2.1. Data Preparation and Indexing

| Component          | Detail                                                            |
|--------------------|-------------------------------------------------------------------|
| **RAG Implementation Dataset** | SciQ (from Allen AI), 13,679 entries of (Question, Support Context, Correct Answer) |
| **Chunking/Indexing** | Combined Question + Context Text<br>Each Q&A pair is concatenated (Question + Context) and embedded to capture complete knowledge units. |
| **Embedding Model** | all-MiniLM-L6-v2 - A fast, 384-dimensional Sentence Transformer optimized for speed on CPU environments. |
| **Persistence Caching** | Raw data (`sciq_data.pkl`) and pre-computed embeddings (`sciq_embeddings.pkl`) saved locally to eliminate re-downloading/re-embedding on subsequent runs. |

### 2.2. Retrieval Component (Search)

- **Typo Correction:** Incoming user queries are first processed by a spellchecker utility (`_correct_spelling`) to fix common scientific typos.  
- **Query Embedding:** The corrected query is encoded into a high-dimensional vector using the all-MiniLM-L6-v2 model.  
- **Similarity Search:** Cosine similarity is computed between the query vector and all 13,679 pre-computed vectors.  
- **Top-K Filtering:** The top *K=5* most similar documents are retrieved; a relevance threshold of *0.5* filters out poor matches, ensuring only highly relevant contexts reach the LLM.

### 2.3. Generation Component (Synthesis)

- **Prompt Construction:** Top 2 retrieved context blocks are concatenated and inserted into a precise system prompt instructing the Qwen LLM to act as a concise science tutor.  
- **LLM:** *Qwen-2.5-0.5B-Instruct* loaded via Hugging Face transformers explicitly on CPU (torch_dtype=torch.float32) for compatibility without a dedicated GPU.  
- **Inference:** Final answers are generated with constraints (`max_new_tokens=60`, `temperature=0.3` for factual output), grounded on retrieved context.

---

## 3. Technical Implementation Details

### 3.1. GUI and Responsiveness

- **Threading:** The `send_message` function runs heavy processing (`process_query`) on a separate worker thread to keep the GUI responsive.  
- **after() Method:** GUI updates (final answers, buttons) are scheduled safely on the main Tkinter thread using `self.window.after()`.  
- **Loading Status:** A `LoadingWindow` provides visual feedback during resource-intensive model loading.

### 3.2. Error Handling and Degradation

| Failure Point      | Solution                                      | Outcome                               |
|--------------------|-----------------------------------------------|-------------------------------------|
| LLM Load Failure   | `_load_llm` uses try-except; sets `self.llm_model=None` | Falls back to Retrieval-Only Mode; summarizes top retrieved contents. |
| Embedding Mismatch | Checks cached embedding dims vs expected dims; regenerates if mismatch | Ensures compatibility of embeddings. |
| No Relevant Context| Enforced retrieval threshold (0.5)           | Returns “No specific context available” prompt to LLM to avoid hallucination. |

---

## 4. Installation and Usage

### 4.1. Requirements

- Python 3.8+  
- Dependencies:  
  - `transformers`  
  - `sentence-transformers`  
  - `datasets`  
  - `scikit-learn`  
  - `pyspellchecker`  
  - `torch`  
  - `numpy`

### 4.2. Setup and Execution

pip install -r requirements.txt
python rag_science_chatbot.py

text

- On first run, the application downloads and caches datasets/models, then generates the vector index. This takes about 10–15 minutes.  
- Subsequent runs start instantly.

### 4.3. Demo Examples

| Query                      | Corrected Query               | RAG Output (LLM)                                                                                     | Best Source Relevance |
|----------------------------|------------------------------|-----------------------------------------------------------------------------------------------------|----------------------|
| "What is mitrochondira?" (typo) | "What is mitochondria?"        | Mitochondria are the organelles within cells that generate most of the cell’s ATP, used as energy. | 78%                  |
| "how does photosythesis work"    | "how does photosynthesis work" | Photosynthesis converts light energy into chemical energy used to fuel organism activities.        | 85%                  |
| "who created newton's laws"       | (No correction needed)        | Sir Isaac Newton developed the three laws of motion describing force and motion relationships.     | 72%                  |

---

## 5. Future Improvements

- **Knowledge Expansion:** Integrate additional domain datasets (e.g., medical, advanced chemistry) to broaden coverage.  
- **LLM Optimization:** Use quantization (e.g., 4-bit loading) with libraries like `bitsandbytes` to reduce memory and speed up inference.  
- **Model Swapping:** Explore smaller, performant models (e.g., Phi-3) optimized for CPU.  
- **Advanced RAG Techniques:** Add re-ranking (e.g., cross-encoder `bge-reranker`) to refine top-K context selection. Implement context compression to maximize LLM window usage.  
- **UI Enhancements:** Add rich text formatting (bold/italics) and speech-to-text voice input for accessibility.

Thank You!