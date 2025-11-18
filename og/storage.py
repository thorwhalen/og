"""Storage layer for OG (Own Ghost) observations.

This module provides storage backends for observations using dol,
with fallback to simple implementations when dol is not available.
"""

import json
import os
import pickle
import tempfile
from collections.abc import Iterator, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from og.base import Observation

# Optional dol import with fallback
try:
    from dol import Files, wrap_kvs, add_ipython_key_completions

    HAVE_DOL = True
except ImportError:
    HAVE_DOL = False

    def add_ipython_key_completions(obj):
        """Fallback: no-op decorator."""
        return obj


class SimpleFileStore(MutableMapping):
    """Simple file-based storage for observations.

    This is a fallback implementation when dol is not available.
    """

    def __init__(self, rootdir: str, extension: str = 'json'):
        """Initialize the file store.

        Args:
            rootdir: Root directory for storage
            extension: File extension (json, pkl)
        """
        self.rootdir = Path(rootdir)
        self.extension = extension
        self.rootdir.mkdir(parents=True, exist_ok=True)

    def _filepath(self, key: str) -> Path:
        """Get filepath for a key."""
        return self.rootdir / f"{key}.{self.extension}"

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        filepath = self._filepath(key)

        if not filepath.exists():
            raise KeyError(key)

        with open(filepath, 'rb') as f:
            data = f.read()

        if self.extension == 'pkl':
            return pickle.loads(data)
        elif self.extension == 'json':
            return json.loads(data.decode())
        else:
            return data.decode()

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key."""
        filepath = self._filepath(key)

        if self.extension == 'pkl':
            data = pickle.dumps(value)
        elif self.extension == 'json':
            data = json.dumps(value).encode()
        else:
            data = str(value).encode()

        with open(filepath, 'wb') as f:
            f.write(data)

    def __delitem__(self, key: str) -> None:
        """Delete item by key."""
        filepath = self._filepath(key)
        if filepath.exists():
            filepath.unlink()
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        for filepath in self.rootdir.glob(f'*.{self.extension}'):
            yield filepath.stem

    def __len__(self) -> int:
        """Get number of items."""
        return sum(1 for _ in self)


class ObservationStore(MutableMapping):
    """Specialized store for Observation objects.

    This wraps a basic file store and handles serialization/deserialization
    of Observation objects.
    """

    def __init__(self, base_store: MutableMapping):
        """Initialize observation store.

        Args:
            base_store: Underlying storage (e.g., SimpleFileStore)
        """
        self.base_store = base_store

    def __getitem__(self, key: str) -> Observation:
        """Get observation by key."""
        data = self.base_store[key]
        if isinstance(data, dict):
            return Observation.from_dict(data)
        return data

    def __setitem__(self, key: str, value: Observation) -> None:
        """Store observation."""
        if isinstance(value, Observation):
            self.base_store[key] = value.to_dict()
        else:
            raise TypeError(f"Expected Observation, got {type(value)}")

    def __delitem__(self, key: str) -> None:
        """Delete observation."""
        del self.base_store[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.base_store)

    def __len__(self) -> int:
        """Get number of observations."""
        return len(self.base_store)


def mk_observation_store(
    rootdir: str, *, extension: str = 'json', use_dol: bool = True
) -> ObservationStore:
    """Create an observation store.

    Args:
        rootdir: Root directory for storage
        extension: File extension (json, pkl)
        use_dol: Whether to use dol if available

    Returns:
        ObservationStore instance
    """
    if HAVE_DOL and use_dol:
        # Use dol for full functionality
        base_store = Files(rootdir)

        # Set up codec based on extension
        if extension == 'pkl':
            encode = pickle.dumps
            decode = pickle.loads
        elif extension == 'json':
            encode = lambda x: json.dumps(x).encode()
            decode = lambda x: json.loads(x.decode())
        else:
            encode = lambda x: str(x).encode()
            decode = lambda x: x.decode()

        # Key transformations to add/remove extension
        def _add_ext(k: str) -> str:
            return f"{k}.{extension}"

        def _remove_ext(k: str) -> str:
            return k.rsplit('.', 1)[0] if '.' in k else k

        store = wrap_kvs(
            base_store,
            key_of_id=_add_ext,
            id_of_key=_remove_ext,
            obj_of_data=decode,
            data_of_obj=encode,
        )

        base = add_ipython_key_completions(store)
    else:
        # Use simple fallback
        base = SimpleFileStore(rootdir, extension)

    return ObservationStore(base)


class ObservationMall:
    """A 'mall' (store of stores) for different types of observations.

    This organizes observations by observer type or category,
    making it easy to query specific types of observations.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """Initialize the mall.

        Args:
            root_dir: Root directory for all stores (uses temp if None)
        """
        if root_dir is None:
            root_dir = os.path.join(tempfile.gettempdir(), 'og_observations')

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._stores: dict[str, ObservationStore] = {}

    def get_store(self, name: str) -> ObservationStore:
        """Get or create a store for a specific observer/category.

        Args:
            name: Name of the observer or category

        Returns:
            ObservationStore for that observer/category
        """
        if name not in self._stores:
            store_dir = self.root_dir / name
            self._stores[name] = mk_observation_store(str(store_dir))

        return self._stores[name]

    def __getitem__(self, name: str) -> ObservationStore:
        """Get store by name (dict-like access)."""
        return self.get_store(name)

    def list_stores(self) -> list[str]:
        """List all available stores.

        Returns:
            List of store names
        """
        # List both in-memory stores and directories on disk
        disk_stores = {
            p.name for p in self.root_dir.iterdir() if p.is_dir()
        }
        return sorted(set(self._stores.keys()) | disk_stores)

    def add_observation(self, observation: Observation) -> str:
        """Add an observation to the appropriate store.

        Args:
            observation: The observation to store

        Returns:
            Key used to store the observation
        """
        # Use observer name to determine which store
        store = self.get_store(observation.observer_name)

        # Generate key from timestamp and event type
        key = f"{observation.timestamp.isoformat()}_{observation.event_type}"

        store[key] = observation
        return key

    def query_by_timerange(
        self,
        start: datetime,
        end: datetime,
        observer_name: Optional[str] = None,
    ) -> list[Observation]:
        """Query observations by time range.

        Args:
            start: Start datetime
            end: End datetime
            observer_name: Optional observer name to filter by

        Returns:
            List of observations in the time range
        """
        observations = []

        stores_to_query = (
            [observer_name] if observer_name else self.list_stores()
        )

        for store_name in stores_to_query:
            try:
                store = self.get_store(store_name)
                for key in store:
                    obs = store[key]
                    if start <= obs.timestamp <= end:
                        observations.append(obs)
            except Exception as e:
                print(f"Error querying store {store_name}: {e}")

        return sorted(observations, key=lambda x: x.timestamp)

    def query_by_event_type(
        self, event_type: str, observer_name: Optional[str] = None
    ) -> list[Observation]:
        """Query observations by event type.

        Args:
            event_type: Event type to filter by
            observer_name: Optional observer name to filter by

        Returns:
            List of matching observations
        """
        observations = []

        stores_to_query = (
            [observer_name] if observer_name else self.list_stores()
        )

        for store_name in stores_to_query:
            try:
                store = self.get_store(store_name)
                for key in store:
                    obs = store[key]
                    if obs.event_type == event_type:
                        observations.append(obs)
            except Exception as e:
                print(f"Error querying store {store_name}: {e}")

        return observations
