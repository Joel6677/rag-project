# RAG System for Superstore Sales Dataset
A Retrieval-Augmented Generation (RAG) system that answers natural language questions about US retail sales data from 2014–2017 using a local LLM.

## Requirements

Python 3.10+
Ollama installed and running locally

## Installation
1. Clone the repository and navigate to the project folder
```bash 
cd rag
```
2. Install Python dependencies
```bash 
pip install -r requirements.txt
```
3. Download the dataset (if needed)
   
- Download the Superstore Sales dataset from Kaggle:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
- Place the file Sample - Superstore.csv in the project root folder.

4. Install and start Ollama
```bash 
# Download Ollama from https://ollama.com then run:
ollama pull phi3
ollama serve
```
Usage
First run — builds the database and runs all queries:
```bash 
python main.py
```
This will:

- Load and prepare the dataset
- Create 10,027 text documents
- Embed and store them in ChromaDB
- Run example queries using the RAG pipeline

Subsequent runs — skip rebuilding the database:
Set rebuild = False in the main() function to load the existing ChromaDB collection instead of rebuilding it from scratch. This makes subsequent runs much faster.
```python
# In main():
rebuild = False  # change this line
```
