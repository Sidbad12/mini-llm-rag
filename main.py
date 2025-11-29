import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import scrolledtext, ttk

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker  

# For LLM Integration
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAGWithLLM:
    """
    Hybrid RAG System:
    - Retrieves relevant context from SciQ dataset
    - Uses Qwen-0.5B to generate natural, contextual answers
    """
    
    def __init__(self):
        self.dataset_path = "sciq_data.pkl"
        self.embeddings_path = "sciq_embeddings.pkl"
        self.retrieval_model_name = "all-MiniLM-L6-v2"  
        self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        self.retrieval_model = None
        self.llm_model = None
        self.tokenizer = None
        self.spell = SpellChecker()  
        
        self.questions = []
        self.contexts = []
        self.answers = []
        self.embeddings = None
        
        print("=" * 60)
        print(" Initializing RAG + LLM Science Chatbot")
        print("=" * 60)
        self.setup()
    
    def setup(self):
        print("\n[1/3] Loading Dataset & Embeddings...")
        self._setup_dataset()
        
        print("\n[2/3] Loading Retrieval Model...")
        self._load_retrieval_model()
        
        print("\n[3/3] Loading LLM (Qwen-0.5B)...")
        self._load_llm()
        
        print("\n All systems ready!\n")
    
    def _setup_dataset(self):
        try:
            if os.path.exists(self.dataset_path) and os.path.exists(self.embeddings_path):
                print("   → Loading cached dataset...")
                self._load_cache()
            else:
                print("   → Downloading SciQ dataset...")
                self._download_and_prepare()
        except Exception as e:
            print(f"    Cache error: {e}")
            print("   → Rebuilding dataset...")
            self._download_and_prepare()
    
    def _load_cache(self):
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)
            self.questions = data["questions"]
            self.contexts = data["contexts"]
            self.answers = data["answers"]
        
        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        
        print(f"    Loaded {len(self.questions):,} questions")
    
    def _download_and_prepare(self):
        dataset = load_dataset("allenai/sciq", split="train")
        
        self.questions = []
        self.contexts = []
        self.answers = []
        
        for item in dataset:
            self.questions.append(item["question"])
            self.contexts.append(item["support"] or "")
            self.answers.append(item["correct_answer"])
        
        # Save dataset
        with open(self.dataset_path, "wb") as f:
            pickle.dump({
                "questions": self.questions,
                "contexts": self.contexts,
                "answers": self.answers
            }, f)
        
        print(f"    Processed {len(self.questions):,} questions")
        print("   → Generating embeddings (one-time)...")
        
        # Generate embeddings
        temp_model = SentenceTransformer(self.retrieval_model_name)
        combined = [f"{q} {c}" for q, c in zip(self.questions, self.contexts)]
        self.embeddings = temp_model.encode(combined, show_progress_bar=True)
        
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        
        print("    Embeddings saved")
    
    def _load_retrieval_model(self):
        self.retrieval_model = SentenceTransformer(self.retrieval_model_name)
        print("    Retrieval model loaded")
        
        if self.embeddings is not None:
            expected_dim = self.retrieval_model.get_sentence_embedding_dimension()
            if self.embeddings.shape[1] != expected_dim:
                print(f"     Embedding dimension mismatch: cached {self.embeddings.shape[1]} vs expected {expected_dim}")
                print("   → Regenerating embeddings with new model...")
                self._regenerate_embeddings()
    
    def _regenerate_embeddings(self):
        combined = [f"{q} {c}" for q, c in zip(self.questions, self.contexts)]
        self.embeddings = self.retrieval_model.encode(combined, show_progress_bar=True)
        
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        
        print("    Embeddings regenerated and saved")
    
    def _load_llm(self):
        try:
            print("   → Downloading Qwen-0.5B (first time only)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float32,  # CPU compatible
                trust_remote_code=True
            )
            self.llm_model.to("cpu") 
            
            print("    Qwen-0.5B loaded successfully")
            
        except Exception as e:
            print(f"     LLM loading failed: {e}")
            print("   → Falling back to retrieval-only mode")
            self.llm_model = None

    def _correct_spelling(self, query):
        words = query.split()
        corrected = [self.spell.correction(word) for word in words]
        return " ".join(corrected)
    
    def retrieve_context(self, query, top_k=5):  
        corrected_query = self._correct_spelling(query)
        print(f"   → Corrected query: '{query}' → '{corrected_query}'")
       
        query_embedding = self.retrieval_model.encode([corrected_query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.5:  
                results.append({
                    "question": self.questions[idx],
                    "context": self.contexts[idx],
                    "answer": self.answers[idx],
                    "score": float(similarities[idx])
                })
        
        return results
    def generate_answer(self, query, context_items):
        
        if not self.llm_model:
            if context_items:
                best = context_items[0]
                return f"Based on retrieved knowledge: {best['context'][:200]}... Answer: {best['answer']}"
            return "I don't have enough information to answer this question."
        
        context_text = ""
        for i, item in enumerate(context_items[:2], 1): 
            if item['context']:
                context_text += f"Context {i}: {item['context']}\n"
                context_text += f"Related answer: {item['answer']}\n\n"

        prompt = f"""You are a science tutor. Answer the student's question using the provided context.
Be clear, accurate, and educational. Keep your answer concise (1-2 sentences). Do not include raw data, JSON, or unrelated details. If the context isn't relevant, say you don't know.

Context:
{context_text if context_text else "No specific context available."}

Student Question: {query}

Answer:"""
        
        # Generate with Qwen
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,  
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                            skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            # Improved fallback
            if context_items:
                return f"{context_items[0]['context'][:200]}... Answer: {context_items[0]['answer']}"
            return "I encountered an error generating the answer."
 
    def answer_question(self, query):
        """Main function: Retrieve + Generate"""
        
        # Step 1: Retrieve relevant context
        print(f"\n🔍 Searching knowledge base for: {query}")
        context_items = self.retrieve_context(query, top_k=5)
        
        if not context_items:
            print("   → No relevant context found")
        else:
            print(f"   → Found {len(context_items)} relevant items")
            print(f"   → Best match score: {context_items[0]['score']:.3f}")
        
        # Step 2: Generate answer using LLM
        print(" Generating answer...")
        answer = self.generate_answer(query, context_items)
        
        # Format response
        response = {
            "answer": answer,
            "sources": context_items[:2] if context_items else [],
            "has_context": len(context_items) > 0
        }
        
        return response

class ChatbotGUI:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.window = tk.Tk()
        self.window.title("Offline Science AI Chatbot - RAG + LLM")
        self.window.geometry("1000x750")
        self.window.configure(bg='#f0f0f0')
        
        self.is_processing = False
        self.setup_ui()
    
    def setup_ui(self):
        """Create GUI"""

        header = tk.Frame(self.window, bg='#1a1a2e', height=120)
        header.pack(fill='x')
        
        title = tk.Label(header, 
                        text="AI Science Chatbot",
                        font=('Arial', 28, 'bold'), 
                        bg='#1a1a2e', 
                        fg='#00ff88')
        title.pack(pady=10)
        
        subtitle = tk.Label(header,
                           text="RAG + Qwen 0.5B | Retrieval-Augmented Generation | Offline Mode",
                           font=('Arial', 11), 
                           bg='#1a1a2e', 
                           fg='#ffffff')
        subtitle.pack()
        
        tech = tk.Label(header,
                       text="💡 Sentence-Transformers + LLM + 13,679 Science Q&A + Spelling Correction",
                       font=('Arial', 9), 
                       bg='#1a1a2e', 
                       fg='#aaaaaa')
        tech.pack(pady=5)
        
        chat_frame = tk.Frame(self.window, bg='white')
        chat_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg='#ffffff',
            fg='#1a1a1a',
            padx=15,
            pady=15,
            relief='flat'
        )
        self.chat_display.pack(fill='both', expand=True)
        self.chat_display.config(state='disabled')
        
        # Configure tags for formatting
        self.chat_display.tag_config('user', foreground='#0066cc', font=('Arial', 11, 'bold'))
        self.chat_display.tag_config('bot', foreground='#00aa44', font=('Arial', 11, 'bold'))
        self.chat_display.tag_config('time', foreground='#888888', font=('Arial', 9))
        self.chat_display.tag_config('source', foreground='#666666', font=('Arial', 9, 'italic'))
        
        # Welcome message
        self.add_message("Bot", 
                        "👋 Hello! I'm your AI science tutor powered by RAG + LLM!\n\n"
                        "I can answer questions about:\n"
                        "• Physics (Newton's laws, energy, motion)\n"
                        "• Chemistry (atoms, molecules, reactions)\n"
                        "• Biology (cells, DNA, photosynthesis)\n\n"
                        "Ask me anything!")
        
        # Input Frame
        input_frame = tk.Frame(self.window, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=15, pady=15)
        
        self.input_field = tk.Entry(
            input_frame,
            font=('Segoe UI', 13),
            bg='white',
            fg='#1a1a1a',
            relief='solid',
            bd=1
        )
        self.input_field.pack(side='left', fill='x', expand=True, ipady=8)
        self.input_field.bind('<Return>', lambda e: self.send_message())
        
        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            font=('Arial', 12, 'bold'),
            bg='#00ff88',
            fg='#1a1a2e',
            padx=25,
            pady=8,
            relief='flat',
            cursor='hand2'
        )
        self.send_btn.pack(side='right', padx=(10, 0))
   
        self.status = tk.Label(
            self.window,
            text="Ready",
            font=('Arial', 5),
            bg='#e0e0e0',
            fg='#555555',
            anchor='w',
            padx=10
        )
        self.status.pack(fill='x', side='bottom')
    
    def add_message(self, sender, text, sources=None):
        self.chat_display.config(state='normal')
        
        timestamp = datetime.now().strftime("%H:%M")
        
        if sender == "You":
            self.chat_display.insert('end', f"\n[{timestamp}] ", 'time')
            self.chat_display.insert('end', f"{sender}:\n", 'user')
        else:
            self.chat_display.insert('end', f"\n[{timestamp}] ", 'time')
            self.chat_display.insert('end', f"{sender}:\n", 'bot')
        
        self.chat_display.insert('end', text + "\n")
    
        if sources:
            self.chat_display.insert('end', "\n📚 Sources:\n", 'source')
            for i, src in enumerate(sources, 1):
                self.chat_display.insert('end', 
                    f"{i}. {src['question']} (relevance: {src['score']:.0%})\n", 
                    'source')
        
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
    
    def send_message(self):
        if self.is_processing:
            return
        
        query = self.input_field.get().strip()
        if not query:
            return
        
        self.input_field.delete(0, 'end')
        self.add_message("You", query)
        
        self.is_processing = True
        self.send_btn.config(state='disabled', text="Processing...")
        self.status.config(text="🔄 Thinking...")
        self.window.after(100, lambda: self.process_query(query))
    
    def process_query(self, query):
        try:
            response = self.rag.answer_question(query)
            self.add_message("Bot", 
                           response["answer"], 
                           sources=response["sources"])
            self.status.config(text="✓ Ready")
            
        except Exception as e:
            self.add_message("Bot", f"❌ Error: {str(e)}")
            self.status.config(text="⚠ Error occurred")
        
        finally:
            self.is_processing = False
            self.send_btn.config(state='normal', text="Send")
    
    def run(self):
        """Start GUI"""
        self.window.mainloop()


def main():
    print("\n" + "="*60)
    print("   OFFLINE SCIENCE AI CHATBOT - RAG + LLM")
    print("   AIML Internship Assignment")
    print("="*60 + "\n")
    
    # Initialize system
    rag_system = RAGWithLLM()
    
    # Launch GUI
    print("Launching GUI...\n")
    gui = ChatbotGUI(rag_system)
    gui.run()


if __name__ == "__main__":
    main()
