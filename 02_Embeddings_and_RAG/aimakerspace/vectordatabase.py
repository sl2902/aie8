import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the euclidean distance between two vectors"""
    # We return negative score, so higher distance = more similar documents
    distance = np.linalg.norm(vector_a - vector_b)
    # Convert distance to similarity: smaller distance = higher similarity
    return 1 / (1 + distance)

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors"""
    distance = np.sum(np.abs(vector_a - vector_b))
    # Convert distance to similarity: smaller distance = higher similarity
    return 1 / (1 + distance)



class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
        else:
            # default
            self.metadata[key] = {
                "timestamp": datetime.now().isoformat(),
                "source": "unknown",
                "chunk_index": len(self.vectors) - 1
            }

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        
        # Filter vectors based on metadata if provided
        if metadata_filter:
            filtered_items = []
            for key, vector in self.vectors.items():
                metadata = self.metadata[key]
                if all(metadata.get(filter_key) == filter_value 
                      for filter_key, filter_value in metadata_filter.items()):
                    filtered_items.append((key, vector))
        else:
            filtered_items = list(self.vectors.items())

        # Calculate similarities using filtered_items, not self.vectors
        scores = [
            (key, distance_measure(query_vector, vector), self.metadata[key])
            for key, vector in filtered_items  # Fixed: use filtered_items and add metadata
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        metadata_filter: Dict[str, Any] = None,
        include_metadata: bool = True,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, metadata_filter)
        
        if return_as_text:
            return [result[0] for result in results]
        elif include_metadata:
            return results  # This now returns (text, score, metadata) tuples
        else:
            return [(result[0], result[1]) for result in results]

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        return self.vectors.get(key, None), self.metadata.get(key, {})

    async def abuild_from_list(self, list_of_text: List[str], source_name: str = "default", source_type: str = "text") -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)

        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = {
                "source": source_name,
                "source_type": source_type,
                "chunk_index": i,
                "chunk_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "total_chunks": len(list_of_text)
            }
            self.insert(text, np.array(embedding), metadata)
        return self
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        if not self.metadata:
            return {}
        
        sources = set()
        source_types = set()
        chunk_counts = {}
        total_chunks = len(self.metadata)
        
        for metadata in self.metadata.values():
            source = metadata.get("source", "unknown")
            sources.add(source)
            source_type = metadata.get("source_type", "unkown")
            source_types.add(source_type)
            chunk_counts[source] = chunk_counts.get(source, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "sources": list(sources),
            "source_types": list(source_types),
            "chunks_per_source": chunk_counts,
            "sample_metadata": next(iter(self.metadata.values())) if self.metadata else {}
        }


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)

    searched_vector = vector_db.search_by_text(
        "I think fruit is awesome!", 
        k=k, 
        distance_measure=euclidean_distance
    )
    print(f"Closest {k} vector(s):", searched_vector)
