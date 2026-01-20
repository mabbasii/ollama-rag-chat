#!/usr/bin/env python3
"""
Vector Database Builder
Reads data from CSV and creates a ChromaDB vector store for RAG retrieval.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
import argparse

def build_vector_db(
    csv_path: str,
    db_path: str = "./chroma_db",
    collection_name: str = "knowledge_base",
    content_column: str = "content",
    metadata_columns: list = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
):
    """
    Build a vector database from a CSV file.

    Args:
        csv_path: Path to the CSV file
        db_path: Where to store the ChromaDB
        collection_name: Name of the collection
        content_column: Column containing the main text content
        metadata_columns: List of columns to include as metadata
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
    """

    # Read CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows")

    # Initialize embedding model
    print("\nLoading embedding model...")
    print("   Model: BAAI/bge-large-en-v1.5")
    print("   Device: CPU")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding model loaded\n")

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Process documents
    documents = []
    skipped = 0

    print("Processing documents...")
    for i, row in df.iterrows():
        # Skip rows with missing content
        if pd.isna(row[content_column]) or not str(row[content_column]).strip():
            skipped += 1
            print(f"Warning: Skipping row {i}: no content")
            continue

        # Build metadata
        metadata = {'row_id': i}
        if metadata_columns:
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    metadata[col] = str(row[col])

        doc = Document(
            page_content=str(row[content_column]),
            metadata=metadata
        )
        documents.append(doc)

    print(f"\nProcessed {len(documents)} documents (skipped {skipped})")

    # Split documents
    print(f"\nSplitting documents into chunks...")
    split_docs = text_splitter.split_documents(documents)
    print(f"   Created {len(split_docs)} chunks from {len(documents)} documents")

    # Create unique IDs
    ids = [f"doc_{i}" for i in range(len(split_docs))]

    # Create vector database
    print("\nCreating vector database...")
    print("   (This may take a few minutes)")

    vector_store = Chroma.from_documents(
        documents=split_docs,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=db_path,
        ids=ids
    )

    print(f"\nVector database created at {db_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows processed: {len(documents)}")
    print(f"Total chunks created: {len(split_docs)}")
    print(f"Rows skipped: {skipped}")
    print(f"Database location: {db_path}")
    print(f"Collection name: {collection_name}")
    print("="*60 + "\n")

    # Verification test
    print("Running verification test...\n")
    test_query = "test query"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"Test query: '{test_query}'")
    print(f"Retrieved {len(results)} results")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. {preview}...")

    print("\nVector database ready!")

    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector database from CSV")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--db-path", default="./chroma_db", help="Output database path")
    parser.add_argument("--collection", default="knowledge_base", help="Collection name")
    parser.add_argument("--content-column", default="content", help="Column with main content")
    parser.add_argument("--metadata", nargs="+", help="Columns to include as metadata")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")

    args = parser.parse_args()

    build_vector_db(
        csv_path=args.csv_path,
        db_path=args.db_path,
        collection_name=args.collection,
        content_column=args.content_column,
        metadata_columns=args.metadata,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
