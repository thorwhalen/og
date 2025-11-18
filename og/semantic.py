"""Semantic search using vector embeddings for OG observations.

This module provides semantic search capabilities over observations,
allowing users to find relevant activity even without exact keyword matches.
"""

import os
from datetime import datetime
from typing import Optional

from og.base import Observation


class SemanticMemory:
    """Vector database for semantic search over observations.

    Uses ChromaDB to store and search observations semantically.
    This allows queries like "machine learning projects" to find
    related activity even without exact keyword matches.
    """

    def __init__(
        self,
        collection_name: str = 'og_observations',
        persist_directory: Optional[str] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
    ):
        """Initialize semantic memory.

        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Lazy import to avoid requiring chromadb if not used
        try:
            import chromadb
            from chromadb.config import Settings

            if persist_directory:
                self.client = chromadb.Client(
                    Settings(
                        persist_directory=persist_directory,
                        anonymized_telemetry=False,
                    )
                )
            else:
                self.client = chromadb.Client()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={'description': 'OG observations for semantic search'},
            )

        except ImportError:
            raise ImportError(
                "chromadb is required for semantic search. "
                "Install it with: pip install chromadb"
            )

    def add_observation(self, obs: Observation) -> None:
        """Add an observation to the vector database.

        Args:
            obs: Observation to add
        """
        # Create text representation for embedding
        text = self._observation_to_text(obs)

        # Create unique ID
        obs_id = f"{obs.observer_name}_{obs.timestamp.isoformat()}_{hash(text) % 10000}"

        # Add to collection
        self.collection.add(
            documents=[text],
            metadatas=[self._observation_to_metadata(obs)],
            ids=[obs_id],
        )

    def add_observations(self, observations: list[Observation]) -> None:
        """Add multiple observations in batch.

        Args:
            observations: List of observations to add
        """
        if not observations:
            return

        texts = [self._observation_to_text(obs) for obs in observations]
        metadatas = [self._observation_to_metadata(obs) for obs in observations]
        ids = [
            f"{obs.observer_name}_{obs.timestamp.isoformat()}_{hash(self._observation_to_text(obs)) % 10000}"
            for obs in observations
        ]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> dict:
        """Search observations semantically.

        Args:
            query: Search query in natural language
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary with search results

        Example:
            >>> memory = SemanticMemory()
            >>> results = memory.search("machine learning projects", n_results=5)
            >>> for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            ...     print(f"{metadata['timestamp']}: {doc}")
        """
        kwargs = {'query_texts': [query], 'n_results': n_results}

        if filter_metadata:
            kwargs['where'] = filter_metadata

        results = self.collection.query(**kwargs)
        return results

    def search_by_timerange(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        n_results: int = 10,
    ) -> dict:
        """Search observations within a time range.

        Args:
            query: Search query
            start_time: Start of time range
            end_time: End of time range
            n_results: Number of results

        Returns:
            Search results
        """
        # ChromaDB doesn't support date range queries directly,
        # so we'll filter in post-processing
        results = self.search(query, n_results=n_results * 2)

        # Filter by time range
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []

        if results['documents']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            ):
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if start_time <= timestamp <= end_time:
                    filtered_docs.append(doc)
                    filtered_metadatas.append(metadata)
                    filtered_distances.append(distance)

                    if len(filtered_docs) >= n_results:
                        break

        return {
            'documents': [filtered_docs],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances],
        }

    def search_by_observer(
        self, query: str, observer_name: str, n_results: int = 10
    ) -> dict:
        """Search observations from a specific observer.

        Args:
            query: Search query
            observer_name: Name of observer to filter by
            n_results: Number of results

        Returns:
            Search results
        """
        return self.search(
            query, n_results=n_results, filter_metadata={'observer_name': observer_name}
        )

    def find_similar(self, observation: Observation, n_results: int = 5) -> dict:
        """Find observations similar to a given observation.

        Args:
            observation: Reference observation
            n_results: Number of similar observations to find

        Returns:
            Similar observations
        """
        text = self._observation_to_text(observation)
        return self.search(text, n_results=n_results)

    def get_all_observations(self) -> dict:
        """Get all observations from the vector database.

        Returns:
            All observations
        """
        # Get count first
        count = self.collection.count()
        if count == 0:
            return {'documents': [[]], 'metadatas': [[]], 'ids': [[]]}

        # ChromaDB requires a query, so we use a generic one
        return self.collection.get()

    def delete_by_timerange(self, start_time: datetime, end_time: datetime) -> int:
        """Delete observations within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Number of observations deleted
        """
        # Get all observations
        all_obs = self.get_all_observations()

        ids_to_delete = []
        if all_obs['metadatas']:
            for obs_id, metadata in zip(all_obs['ids'], all_obs['metadatas']):
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if start_time <= timestamp <= end_time:
                    ids_to_delete.append(obs_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def clear(self) -> None:
        """Clear all observations from the database."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={'description': 'OG observations for semantic search'},
        )

    def _observation_to_text(self, obs: Observation) -> str:
        """Convert observation to text for embedding.

        Args:
            obs: Observation to convert

        Returns:
            Text representation
        """
        # Create rich text representation
        parts = [
            f"Event: {obs.event_type}",
            f"Observer: {obs.observer_name}",
        ]

        # Add data fields
        if obs.data:
            for key, value in obs.data.items():
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, list) and all(
                    isinstance(x, (str, int, float)) for x in value
                ):
                    parts.append(f"{key}: {', '.join(map(str, value[:5]))}")

        # Add tags
        if obs.tags:
            parts.append(f"Tags: {', '.join(obs.tags)}")

        return ' | '.join(parts)

    def _observation_to_metadata(self, obs: Observation) -> dict:
        """Convert observation to metadata dict.

        Args:
            obs: Observation to convert

        Returns:
            Metadata dictionary
        """
        return {
            'timestamp': obs.timestamp.isoformat(),
            'observer_name': obs.observer_name,
            'event_type': obs.event_type,
            'tags': ','.join(obs.tags) if obs.tags else '',
        }

    def get_stats(self) -> dict:
        """Get statistics about the semantic memory.

        Returns:
            Dictionary with statistics
        """
        count = self.collection.count()

        stats = {
            'total_observations': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
        }

        # Get observation breakdown by type if we have observations
        if count > 0:
            all_obs = self.get_all_observations()
            event_types = {}
            observers = {}

            if all_obs['metadatas']:
                for metadata in all_obs['metadatas']:
                    event_type = metadata.get('event_type', 'unknown')
                    observer = metadata.get('observer_name', 'unknown')

                    event_types[event_type] = event_types.get(event_type, 0) + 1
                    observers[observer] = observers.get(observer, 0) + 1

            stats['by_event_type'] = event_types
            stats['by_observer'] = observers

        return stats
