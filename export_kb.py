"""
One-time script: export ChromaDB knowledge base → knowledge_base.npz
Run locally after ingest.py has built chroma_db/.
"""
import chromadb
import numpy as np
import pathlib

CHROMA_DIR = pathlib.Path("chroma_db")
COLLECTION = "paws_knowledge"

chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
col    = chroma.get_collection(COLLECTION)

result = col.get(include=["embeddings", "documents", "metadatas"])

embeddings = np.array(result["embeddings"], dtype=np.float32)
documents  = np.array(result["documents"], dtype=object)
sources    = np.array([m["source"] for m in result["metadatas"]], dtype=object)

np.savez(
    "knowledge_base.npz",
    embeddings=embeddings,
    documents=documents,
    sources=sources,
)
print(f"Exported {len(result['documents'])} chunks → knowledge_base.npz")
print(f"Embedding matrix: {embeddings.shape}")
