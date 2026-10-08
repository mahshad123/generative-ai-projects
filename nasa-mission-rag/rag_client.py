import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path

# Must match the embedding model used by the embedding pipeline, so query
# vectors and stored document vectors live in the same space.
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # None -> api.openai.com


def _build_embedding_function() -> Optional[OpenAIEmbeddingFunction]:
    """Rebuild the OpenAI embedding function used to embed search queries."""
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=OPENAI_BASE_URL,
        model_name=EMBEDDING_MODEL,
    )

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            for collection in collections:
                key = f"{chroma_dir.name}/{collection.name}"
                try:
                    count = collection.count()
                except Exception:
                    count = "unknown"
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} - {collection.name} ({count} docs)",
                    "count": str(count),
                }
        except Exception as e:
            backends[chroma_dir.name] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} - error: {str(e)[:50]}",
                "count": "0",
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # Re-attach the same OpenAI embedding function the pipeline used, so that
    # query_texts embeds searches into the same vector space as the documents.
    embedding_function = _build_embedding_function()

    return client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None
    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["Retrieved NASA mission documents:\n"]

    for i, (document, metadata) in enumerate(zip(documents, metadatas), 1):
        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source", "unknown")
        category = metadata.get("document_category", metadata.get("category", "general"))
        category = category.replace("_", " ").title()

        source_header = f"[Source {i}] Mission: {mission} | Source: {source} | Category: {category}"
        context_parts.append(source_header)

        max_length = 1000
        if len(document) > max_length:
            document = document[:max_length] + "..."
        context_parts.append(document)

    return "\n".join(context_parts)
