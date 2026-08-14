# 🚀 Startup AI Agent

An AI-powered chatbot designed to help entrepreneurs set up and grow their startups.

## 📌 Problem Statement

Entrepreneurs often need to search multiple sources to find information about government schemes, funding opportunities, investors, startup infrastructure and registration-related information.

The Startup AI Agent provides this information through a conversational chatbot using Retrieval-Augmented Generation (RAG).

## 🎯 Features

- Government scheme information
- Startup funding information
- Investor information
- Startup infrastructure information
- Incubator and accelerator information
- Startup registration and policy information
- PDF-based knowledge retrieval
- Semantic search
- ChromaDB vector database
- Cohere LLM
- Streamlit chatbot interface

## 🧠 Architecture

User
↓
Streamlit Chatbot
↓
Startup AI Agent
↓
Query
↓
ChromaDB
↓
Relevant Knowledge
↓
Cohere LLM
↓
Final Answer

## 🔄 RAG Pipeline

PDF Documents
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
ChromaDB
↓
Similarity Search
↓
Relevant Context
↓
Cohere
↓
Answer

## 🛠️ Technologies

- Python
- Cohere API
- ChromaDB
- Streamlit
- PyPDF
- python-dotenv

## 📂 Knowledge Base

The chatbot uses documents covering:

1. Government schemes
2. Investors and funding
3. Startup infrastructure
4. Startup registration and policies

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/startup-ai-agent.git
cd startup-ai-agent
