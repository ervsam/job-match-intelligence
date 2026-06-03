import json
import chromadb
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/chromadb"
COLLECTION_NAME = "job_postings"


def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection


def load_jobs(embeddings_path="data/job_embeddings.jsonl"):
    """Load job embeddings from disk."""
    jobs = []
    with open(embeddings_path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def ingest(embeddings_path="data/job_embeddings.jsonl"):
    """Ingest all job embeddings into ChromaDB."""
    collection = get_collection()

    # Skip if already ingested
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} items. Skipping ingest.")
        print("Delete data/chromadb to re-ingest from scratch.")
        return collection

    jobs = load_jobs(embeddings_path)
    print(f"Ingesting {len(jobs)} jobs into ChromaDB...")

    collection.add(
        ids=[job["id"] for job in jobs],
        embeddings=[job["embedding"] for job in jobs],
        metadatas=[{
            "subject": job["subject"],
            "company": job["company"],
            "date": job["date"],
        } for job in jobs]
    )

    print(f"Done. Collection now has {collection.count()} items.")
    return collection


def query(profile_embedding: list[float], n_results: int = 20):
    """Query ChromaDB with a profile embedding, return top-N jobs."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[profile_embedding],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    # Zip into a clean list of dicts
    hits = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "subject": metadata["subject"],
            "company": metadata["company"],
            "date": metadata["date"],
            "score": round(1 - distance, 4)  # cosine distance → similarity
        })
    return hits


if __name__ == "__main__":
    # Ingest jobs
    ingest()

    # Load profile embedding and query
    with open("data/profile_embedding.json") as f:
        profile = json.load(f)

    print("\nTop 10 most relevant jobs for your profile:\n")
    hits = query(profile["embedding"], n_results=10)
    for i, hit in enumerate(hits, 1):
        print(f"{i:>2}. [{hit['score']:.4f}]  {hit['company'] or '?':<25}  {hit['subject'][:60]}")
