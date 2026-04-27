import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


def load_csv():
    df = pd.read_csv('Sample - Superstore.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter

    return df


def create_row_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Order {row['Order ID']} on {row['Order Date'].strftime('%Y-%m-%d')}: "
            f"Customer {row['Customer Name']} in {row['City']}, {row['State']} ({row['Region']} region) "
            f"bought {row['Quantity']} unit(s) of '{row['Product Name']}' "
            f"(Category: {row['Category']}, Sub-Category: {row['Sub-Category']}) "
            f"for ${row['Sales']:.2f} with a discount of {row['Discount']*100:.0f}%. "
            f"Profit was ${row['Profit']:.2f}."
        )
        metadata = {
            "chunk_type": "row",
            "region": row["Region"],
            "category": row["Category"],
            "sub_category": row["Sub-Category"],
            "year": int(row["Year"]),
            "quarter": int(row["Quarter"]),
            "month": int(row["Month"]),
        }
        documents.append({"text": text, "metadata": metadata})

    return documents


def create_aggregated_documents(df):
    documents = []

    # Yearly summaries
    for year, group in df.groupby("Year"):
        text = (
            f"Year {year} summary: Total sales were ${group['Sales'].sum():,.2f}, "
            f"total profit was ${group['Profit'].sum():,.2f}, "
            f"with {len(group)} orders across all regions and categories."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "yearly_summary", "year": int(year)}})
    # Monthly summaries
    for month, group in df.groupby("Month"):
        text = (
            f"Month {month} summary across all years: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, {len(group)} orders."
        )
        documents.append({"text": text, "metadata": {
            "chunk_type": "monthly_summary", "month": int(month)}})

    # Sub-category summaries
    for subcat, group in df.groupby("Sub-Category"):
        text = (
            f"{subcat} sub-category summary: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, "
            f"{len(group)} orders, average discount {group['Discount'].mean()*100:.1f}%."
        )
        documents.append({"text": text, "metadata": {
            "chunk_type": "subcategory_summary", "sub_category": subcat}})
    # Regional summaries
    for region, group in df.groupby("Region"):
        text = (
            f"{region} region summary: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, "
            f"{len(group)} orders, average discount {group['Discount'].mean()*100:.1f}%."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "regional_summary", "region": region}})

    # Category summaries
    for category, group in df.groupby("Category"):
        text = (
            f"{category} category summary: Total sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, "
            f"{len(group)} orders, average discount {group['Discount'].mean()*100:.1f}%."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "category_summary", "category": category}})

    # Region + Year summaries
    for (region, year), group in df.groupby(["Region", "Year"]):
        text = (
            f"{region} region in {year}: Sales ${group['Sales'].sum():,.2f}, "
            f"profit ${group['Profit'].sum():,.2f}, {len(group)} orders."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "region_year_summary", "region": region, "year": int(year)}})

    return documents


def create_statistical_documents(df):
    documents = []

    # Top performing sub-categories
    top_subs = df.groupby(
        "Sub-Category")["Profit"].sum().sort_values(ascending=False)
    top_text = "Top 5 sub-categories by profit: " + ", ".join(
        [f"{sub} (${profit:,.2f})" for sub, profit in top_subs.head(5).items()]
    )
    documents.append({"text": top_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    # Bottom performing sub-categories
    bottom_text = "Bottom 5 sub-categories by profit: " + ", ".join(
        [f"{sub} (${profit:,.2f})" for sub, profit in top_subs.tail(5).items()]
    )
    documents.append({"text": bottom_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    # Profit margin by category
    for category, group in df.groupby("Category"):
        margin = (group["Profit"].sum() / group["Sales"].sum()) * 100
        text = (
            f"{category} profit margin: {margin:.1f}%. "
            f"Mean sale value ${group['Sales'].mean():,.2f}, "
            f"median profit ${group['Profit'].median():,.2f}."
        )
        documents.append({"text": text, "metadata": {
                         "chunk_type": "statistical_summary", "category": category}})

    # Most discounted sub-categories
    top_discount = df.groupby(
        "Sub-Category")["Discount"].mean().sort_values(ascending=False)
    discount_text = "Most discounted sub-categories: " + ", ".join(
        [f"{sub} ({disc*100:.1f}%)" for sub,
         disc in top_discount.head(5).items()]
    )
    documents.append({"text": discount_text, "metadata": {
                     "chunk_type": "statistical_summary"}})

    return documents


def create_text_documents(df):
    row_docs = create_row_documents(df)
    agg_docs = create_aggregated_documents(df)
    stat_docs = create_statistical_documents(df)
    all_docs = row_docs + agg_docs + stat_docs
    print(f"Total documents: {len(all_docs)}")
    return all_docs


def chunk_documents(all_docs, chunk_size=1000, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunked_docs = []

    for doc in all_docs:
        chunks = splitter.split_text(doc["text"])

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

    return chunked_docs


def create_vector_store(chunks, collection_name="superstore"):
    # Step 1: Create ChromaDB client (stores data locally)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Step 2: Set up the embedding model
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists to avoid duplicate ID errors
    try:
        client.delete_collection(name=collection_name)
        print(f"Cleared existing collection '{collection_name}'")
    except Exception as e:
        print(f"No existing collection to clear: {e}")

    # Step 3: Create a collection (like a table in a database)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn  # type: ignore
    )

    # Step 4: Add chunks in batches
    batch_size = 500
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Stored {min(i + batch_size, total)}/{total} chunks...")

    print(
        f"Done! Collection '{collection_name}' has {collection.count()} chunks")
    return collection


def inspect_vector_store(collection):
    print("\n--- ChromaDB Inspection ---")
    print(f"Total chunks: {collection.count()}")

    # Peek at first 3 chunks
    results = collection.peek(3)
    for i in range(len(results["ids"])):
        print(f"\nChunk {i+1}:")
        print(f"  ID:       {results['ids'][i]}")
        print(f"  Text:     {results['documents'][i]}")
        print(f"  Metadata: {results['metadatas'][i]}")

    # Test metadata filtering
    print("\n--- Metadata Filter Test ---")
    filtered = collection.get(where={"region": "West"})
    print(f"Chunks with region=West: {len(filtered['ids'])}")

    filtered = collection.get(where={"category": "Furniture"})
    print(f"Chunks with category=Furniture: {len(filtered['ids'])}")

    filtered = collection.get(where={"year": 2017})
    print(f"Chunks with year=2017: {len(filtered['ids'])}")


def query_vector_store(collection, query_text, n_results=5, filters=None):
    print("\n--- Query ---")
    print(f"Question: {query_text}")
    if filters:
        print(f"Filters:  {filters}")

    # Similarity search with optional metadata filtering
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=filters if filters else None
    )

    # Format results nicely
    chunks = []
    for i in range(len(results["ids"][0])):
        chunk = {
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        chunks.append(chunk)
        print(f"\nResult {i+1} (distance: {chunk['distance']:.4f}):")
        print(f"  Text:     {chunk['text']}")
        print(f"  Metadata: {chunk['metadata']}")

    return chunks


def generate_answer(query_text, retrieved_chunks, llm, strategy="zero-shot"):

    context = "\n\n".join(
        [f"[Chunk {i+1}]: {chunk['text']}"
         for i, chunk in enumerate(retrieved_chunks)]
    )

    system_message = SystemMessage(content="""
        You are a retail sales analyst for Superstore (2014-2017).
        Use ONLY the data provided in the context below to answer questions.
        Do NOT use any knowledge outside of the provided context.
        Always cite specific numbers from the context in your answer.
        If the context does not contain enough information, say "Insufficient data in context".
    """)

    if strategy == "zero-shot":
        human_message = HumanMessage(content=f"""
Context:
{context}

Question: {query_text}
        """)

    elif strategy == "few-shot":
        human_message = HumanMessage(content=f"""
Context:
{context}

Here is an example of a good answer format:
Q: Which category had the highest sales?
A: Technology had the highest sales with $836,154.03, followed by Furniture 
   with $741,999.80 and Office Supplies with $719,047.03.

Now answer in the same format:
Question: {query_text}
    """)

    elif strategy == "chain-of-thought":
        human_message = HumanMessage(content=f"""
Context:
{context}

Answer the question by following these steps:
Step 1: Identify the relevant data from the context
Step 2: Compare or calculate if needed
Step 3: Conclude with specific numbers

Question: {query_text}
        """)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: zero-shot, few-shot, chain-of-thought")

    response = llm.invoke([system_message, human_message])
    return response.content


def rag_pipeline(collection, query_text, llm, n_results=5, filters=None, strategy="zero-shot"):
    print(f"\n{'='*60}")
    print(f"Query:    {query_text}")
    print(f"Strategy: {strategy}")
    if filters:
        print(f"Filters:  {filters}")
    print('='*60)

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = query_vector_store(
        collection,
        query_text=query_text,
        n_results=n_results,
        filters=filters
    )

    # Step 2: Generate answer
    print("\n--- Generated Answer ---")
    answer = generate_answer(
        query_text, retrieved_chunks, llm=llm, strategy=strategy)
    print(answer)

    return answer


def main():
    df = load_csv()

    all_docs = create_text_documents(df)

    # Only rebuild if needed
    rebuild = False  # set to False to skip rebuilding

    if rebuild:
        # test chunks - only for printing
        for size in [500, 1000, 2000]:
            test_chunks = chunk_documents(all_docs, chunk_size=size)
            print(f"Chunk size {size}: {len(test_chunks)} chunks")

        # Working chunks - used for ChromaDB
        chunks = chunk_documents(all_docs, chunk_size=500)
        print(
            f"\nUsing chunk_size=500: {len(chunks)} chunks ready for ChromaDB")

        lengths = [len(doc["text"]) for doc in all_docs]
        print(f"Shortest document: {min(lengths)} chars")
        print(f"Longest document:  {max(lengths)} chars")
        print(f"Average length:    {sum(lengths)//len(lengths)} chars")

        collection = create_vector_store(chunks)
    else:
        # Load existing collection
        client = chromadb.PersistentClient(path="./chroma_db")
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.get_collection(
            name="superstore",
            embedding_function=embedding_fn  # type: ignore
        )
        print(f"Loaded existing collection with {collection.count()} chunks")

    # inspect_vector_store(collection)

    llm = ChatOllama(model="phi3", temperature=0)

    # 1. Trend analysis — zero-shot
    rag_pipeline(
        collection,
        llm=llm,
        query_text="What were the total sales each year?",
        n_results=5,
        filters={"chunk_type": "yearly_summary"},
        strategy="zero-shot"
    )

    # 2. Category analysis — few-shot
    rag_pipeline(
        collection,
        llm=llm,
        query_text="Which category had the highest profit?",
        n_results=5,
        filters={"chunk_type": "category_summary"},
        strategy="few-shot"
    )

    # 3. Regional analysis — chain-of-thought
    rag_pipeline(
        collection,
        llm=llm,
        query_text="How did the West region perform compared to other regions?",
        n_results=5,
        filters={"chunk_type": "regional_summary"},
        strategy="chain-of-thought"
    )

    # 4. Comparative analysis — chain-of-thought
    rag_pipeline(
        collection,
        llm=llm,
        query_text="Compare sales and profit across all regions in 2017",
        n_results=8,
        filters={"year": 2017},
        strategy="chain-of-thought"
    )

    # 5. Statistical analysis — zero-shot
    rag_pipeline(
        collection,
        llm=llm,
        query_text="What are the most discounted sub-categories?",
        n_results=5,
        filters={"chunk_type": "statistical_summary"},
        strategy="zero-shot"
    )

    # 6.
    rag_pipeline(
        collection,
        llm=llm,
        query_text="Which year had the lowest sales?",
        n_results=5,
        filters={"chunk_type": "yearly_summary"},
        strategy="zero-shot"
    )

    # 7.
    rag_pipeline(
        collection,
        llm=llm,
        query_text="Which region had the highest profit in 2017?",
        n_results=5,
        filters={"year": 2017},
        strategy="zero-shot"
    )

    # "Which month had the highest sales?"
    rag_pipeline(
        collection, llm=llm,
        query_text="Which month had the highest total sales?",
        n_results=12,                                    # all 12 months
        filters={"chunk_type": "monthly_summary"},
        strategy="zero-shot"                             # simple fact
    )

    # "How did Bookcases perform overall?"
    rag_pipeline(
        collection, llm=llm,
        query_text="How did Bookcases perform in terms of sales and profit?",
        n_results=3,
        filters={"chunk_type": "subcategory_summary"},
        strategy="zero-shot"
    )

    # "Compare West and East profit in 2016 vs 2017"
    rag_pipeline(
        collection, llm=llm,
        query_text="Compare West and East region profit in 2016 versus 2017",
        n_results=8,
        # has all region+year combos
        filters={"chunk_type": "region_year_summary"},
        strategy="chain-of-thought"                      # comparison needed
    )

    # "What is the profit margin for Technology?"
    rag_pipeline(
        collection, llm=llm,
        query_text="What is the profit margin for the Technology category?",
        n_results=3,
        filters={
            "$and": [
                {"chunk_type": {"$eq": "statistical_summary"}},
                {"category": {"$eq": "Technology"}}
            ]
        },
        strategy="zero-shot"
    )


if __name__ == "__main__":
    main()
