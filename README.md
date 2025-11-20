# 🧠 Dynamic Memory RAG for Long-Term Conversations  
### Retrieval-Augmented Generation (RAG) System for Conversational Memory

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** architecture designed specifically for *conversational memory*.  
The goal is to understand how different retrieval-based memory mechanisms influence an LLM’s ability to **retain**, **retrieve**, and **update** user-specific information across long multi-turn dialogues.

Unlike knowledge-base RAG systems that retrieve external documents,  
**our system performs RAG over the conversation history itself** —  
creating a *dynamic, evolving memory* of the user.

---

## 🎯 Project Goal

To build a **RAG Memory Module** capable of:

- Maintaining long-term conversational context  
- Retrieving relevant past information  
- Handling fact updates and contradictions  
- Reducing memory decay over long dialogues  
- Comparing multiple retrieval strategies

We evaluate which RAG memory configuration works best under different conversational conditions.

---

## 🔍 Why Conversational RAG?

Standard RAG retrieves external knowledge (web pages, documents).  
However, **in conversation**, the most important context is:

- user preferences  
- personal facts  
- earlier answers  
- past decisions  

Thus, the “knowledge base” becomes the **conversation itself**.

This project implements a **Dynamic Memory RAG System**, where:

- Each message is stored in a memory index  
- A retriever selects relevant past chunks  
- The LLM generates the next answer using retrieved context  

This is *exactly* the RAG principle, applied not to external text but to conversational memory.

---

## 🧩 RAG Memory Architecture (Our System)

