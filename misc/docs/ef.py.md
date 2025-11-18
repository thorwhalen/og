## __init__.py

```python
"""
ef (Embedding Flow) - Lightweight framework for embedding pipelines.

This package provides:
- Easy project creation with flexible storage
- Component registries as mapping stores
- Automatic pipeline composition
- Flexible "out of the box" for simple situations
- Plugin system for adding production implementations

Example:
    >>> from ef import Project
    >>>
    >>> # Create project (works immediately with built-in components)
    >>> project = Project.create('my_project', backend='memory')
    >>>
    >>> # Add data
    >>> project.add_source('doc1', 'Sample text to analyze')
    >>>
    >>> # Create and run pipeline
    >>> _ = project.create_pipeline('analysis', embedder='simple', clusterer='simple_kmeans')
    >>> results = project.run_pipeline('analysis')
    >>>
    >>> # Access persisted results
    >>> len(project.embeddings)
    1
"""

from ef.projects import Project, Projects
from ef.base import (
    ComponentRegistry,
    SegmentKey,
    Segment,
    Vector,
    PlanarVector,
    ClusterIndex,
)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'Project',
    'Projects',
    'ComponentRegistry',
    'SegmentKey',
    'Segment',
    'Vector',
    'PlanarVector',
    'ClusterIndex',
]
```

## base.py

```python
"""
Core types and protocols for ef (Embedding Flow).

This module provides:
- ComponentRegistry: Mapping-based store for pipeline components
- Type aliases for common data types
- Core protocols and interfaces
"""

from collections.abc import Iterator, Callable, MutableMapping
from typing import Any


# ============================================================================
# Type Aliases
# ============================================================================

SegmentKey = str
Segment = str
Vector = list[float]
PlanarVector = tuple[float, float]
ClusterIndex = int


# ============================================================================
# Component Registry
# ============================================================================


class ComponentRegistry(MutableMapping):
    """
    A registry that stores pipeline components (functions) in a Mapping interface.

    This allows components to be accessed like a dictionary while maintaining
    additional metadata about each component.

    >>> registry = ComponentRegistry('embedders')
    >>> registry['simple'] = lambda x: [1.0, 2.0, 3.0]
    >>> 'simple' in registry
    True
    >>> result = registry['simple']("test text")
    """

    def __init__(self, name: str):
        self.name = name
        self._components: dict[str, Callable] = {}
        self._metadata: dict[str, dict] = {}

    def __getitem__(self, key: str) -> Callable:
        return self._components[key]

    def __setitem__(self, key: str, value: Callable) -> None:
        if not callable(value):
            raise TypeError(f"Component must be callable, got {type(value)}")
        self._components[key] = value

    def __delitem__(self, key: str) -> None:
        del self._components[key]
        if key in self._metadata:
            del self._metadata[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def register(
        self, name: str | None = None, **metadata
    ) -> Callable[[Callable], Callable]:
        """
        Decorator to register a component.

        >>> registry = ComponentRegistry('embedders')
        >>> @registry.register('my_embedder', dimension=128)
        ... def embed(text):
        ...     return [1.0] * 128
        >>> 'my_embedder' in registry
        True
        """

        def decorator(func: Callable) -> Callable:
            key = name or func.__name__
            self[key] = func
            if metadata:
                self._metadata[key] = metadata
            return func

        return decorator

    def get_metadata(self, key: str) -> dict:
        """Get metadata for a component."""
        return self._metadata.get(key, {})
```

## dag.py

```python
"""
Pipeline assembly using DAG composition.

This module provides:
- DAG assembly from components
- Simple DAG execution that collects all intermediate results
- Optional future integration with meshed for advanced features
"""

from typing import Callable, Any
from collections.abc import Mapping
import inspect


# ============================================================================
# Function Node
# ============================================================================


class FuncNode:
    """A node in the pipeline DAG representing a function."""

    def __init__(self, func, name=None, bind=None, out=None):
        self.func = func
        self.name = name or func.__name__
        self.bind = bind or {}
        self.out = out or 'result'


# ============================================================================
# Simple Pipeline DAG
# ============================================================================


class DAG:
    """
    Simple DAG implementation for pipeline execution.

    Executes a sequence of functions, collecting all intermediate results.
    Returns a dictionary with all outputs.
    """

    def __init__(self, nodes):
        """
        Initialize DAG.

        Args:
            nodes: List of FuncNode objects or a single FuncNode
        """
        self.nodes = nodes if isinstance(nodes, list) else [nodes]
        self.graph = {node.name: node for node in self.nodes}

    def __call__(self, **kwargs) -> dict:
        """
        Execute the DAG.

        Args:
            **kwargs: Initial inputs to the pipeline

        Returns:
            Dictionary containing all intermediate and final results
        """
        results = dict(kwargs)

        for node in self.nodes:
            # Get inputs for this node
            sig = inspect.signature(node.func)
            func_kwargs = {}

            # Use bind if specified (explicit mapping)
            if node.bind:
                for param, source in node.bind.items():
                    if source in results:
                        func_kwargs[param] = results[source]
            else:
                # Auto-match parameters by name
                for param in sig.parameters:
                    if param in results:
                        func_kwargs[param] = results[param]

            # Execute function
            try:
                output = node.func(**func_kwargs)
                results[node.out] = output
            except TypeError as e:
                # Provide helpful error message
                raise RuntimeError(
                    f"Error calling {node.name}: {e}\n"
                    f"  Available in results: {list(results.keys())}\n"
                    f"  Tried to pass: {list(func_kwargs.keys())}\n"
                    f"  Function signature: {sig}"
                )

        return results


# ============================================================================
# Pipeline Assembly
# ============================================================================


def assemble_pipeline_dag(
    *,
    segmenter: Callable | None = None,
    embedder: Callable | None = None,
    planarizer: Callable | None = None,
    clusterer: Callable | None = None,
) -> DAG:
    """
    Assemble a DAG from pipeline components.

    Components are connected automatically based on their input/output names.

    Args:
        segmenter: Optional segmentation function (source -> segments)
        embedder: Optional embedding function (segments -> embeddings)
        planarizer: Optional planarization function (embeddings -> planar_embeddings)
        clusterer: Optional clustering function (embeddings -> clusters)

    Returns:
        DAG that can be executed with source data

    >>> def seg(source): return {'main': source}
    >>> def emb(segments): return {k: [1.0, 2.0] for k in segments}
    >>> dag = assemble_pipeline_dag(segmenter=seg, embedder=emb)
    >>> results = dag(source='test')
    >>> 'embeddings' in results
    True
    """
    nodes = []

    # Add segmenter if provided
    if segmenter:
        nodes.append(FuncNode(segmenter, name='segment_func', out='segments'))

    # Add embedder if provided (depends on segments)
    if embedder:
        nodes.append(
            FuncNode(
                embedder,
                name='embed_func',
                bind={'segments': 'segments'},  # Connect to segmenter output
                out='embeddings',
            )
        )

    # Add planarizer if provided (depends on embeddings)
    if planarizer:
        nodes.append(
            FuncNode(
                planarizer,
                name='planarize_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='planar_embeddings',
            )
        )

    # Add clusterer if provided (depends on embeddings)
    if clusterer:
        nodes.append(
            FuncNode(
                clusterer,
                name='cluster_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='clusters',
            )
        )

    return DAG(nodes)
```

## plugins/README.md

```python
# Writing ef Plugins

ef plugins extend the framework with additional components (embedders, planarizers, clusterers, segmenters).

## Built-in Plugins

### simple (Built-in)

Toy implementations that work out-of-the-box without dependencies.

```python
from ef import Project
from ef.plugins import simple

project = Project.create('test', backend='memory')
simple.register_simple_components(project)

# Now has toy implementations
print(project.embedders.keys())
# ['simple', 'char_counts']
```

**Components:**
- **Segmenters**: `identity`, `lines`, `sentences`
- **Embedders**: `simple` (char/word/punct counts), `char_counts` (26D letter frequencies)
- **Planarizers**: `simple_2d`, `normalize_2d`
- **Clusterers**: `simple_kmeans`, `threshold`

### imbed (Optional)

Bridge to production imbed package with real ML implementations.

```python
from ef import Project
from ef.plugins import imbed

project = Project.create('production')
imbed.register(project)

# Now has production components
print(project.embedders.keys())
# ['openai-small', 'openai-large', ...]
```

**Requires:** `pip install imbed` or `pip install ef[imbed]`

## Writing Your Own Plugin

Create a new plugin module in `ef/plugins/`:

```python
# ef/plugins/my_plugin.py

def register(project):
    """Register all components from this plugin."""
    _register_embedders(project)
    _register_clusterers(project)


def _register_embedders(project):
    """Add custom embedders."""
    
    @project.embedders.register('my_embedder', dimension=768)
    def my_embedder(segments):
        """My custom embedding implementation."""
        # Your code here
        return {key: compute_embedding(text) for key, text in segments.items()}


def _register_clusterers(project):
    """Add custom clusterers."""
    
    @project.clusterers.register('my_clusterer')
    def my_clusterer(embeddings, *, n_clusters=5):
        """My custom clustering implementation."""
        # Your code here
        return {key: assign_cluster(vec) for key, vec in embeddings.items()}
```

Then use it:

```python
from ef import Project
from ef.plugins import my_plugin

project = Project.create('custom')
my_plugin.register(project)
```

## Plugin Guidelines

1. **Registration Function**: Provide a `register(project)` function
2. **Decorators**: Use `@project.{component_type}.register(name)` to add components
3. **Metadata**: Include metadata in registration (e.g., `dimension=768`)
4. **Signatures**: Follow standard signatures:
   - Segmenter: `(source: str) -> dict[str, str]`
   - Embedder: `(segments: dict[str, str]) -> dict[str, list[float]]`
   - Planarizer: `(embeddings: dict[str, list[float]]) -> dict[str, tuple[float, float]]`
   - Clusterer: `(embeddings: dict[str, list[float]], **kwargs) -> dict[str, int]`
5. **Dependencies**: Import heavy dependencies inside functions (lazy loading)
6. **Error Handling**: Provide clear error messages if dependencies missing

## Example: Sentence Transformers Plugin

```python
# ef/plugins/sentence_transformers_plugin.py

def register(project):
    """Register sentence-transformers models."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: "
            "pip install sentence-transformers"
        )
    
    @project.embedders.register('all-MiniLM-L6-v2', dimension=384)
    def minilm_embedder(segments):
        """Efficient sentence embedder."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        keys = list(segments.keys())
        texts = list(segments.values())
        
        embeddings = model.encode(texts)
        
        return {key: emb.tolist() for key, emb in zip(keys, embeddings)}
```

## Distribution

To distribute your plugin:

1. **Standalone Package**:
   ```bash
   # my_ef_plugin/
   # ├── setup.py
   # └── my_ef_plugin.py
   
   pip install my-ef-plugin
   ```

2. **Include in ef.plugins** (for official plugins):
   - Add to `ef/plugins/`
   - Add to `ef/plugins/__init__.py`
   - Add optional dependency to `setup.cfg`

## Testing Plugins

```python
import pytest
from ef import Project

def test_my_plugin():
    """Test plugin registration."""
    from ef.plugins import my_plugin
    
    project = Project.create('test', backend='memory')
    my_plugin.register(project)
    
    # Check components registered
    assert 'my_embedder' in project.embedders
    
    # Test functionality
    project.add_source('test', 'Sample text')
    results = project.quick_embed('Sample text', embedder='my_embedder')
    
    assert 'main' in results
    assert len(results['main']) == 768  # Check dimension
```
```

## plugins/__init__.py

```python
"""
ef plugins - Extend ef with additional components.

This package provides plugin system for ef:
- simple: Built-in toy implementations (works out-of-the-box)
- imbed: Bridge to production imbed package (requires pip install imbed)

Usage:
    >>> from ef import Project
    >>> from ef.plugins import simple
    >>>
    >>> project = Project.create('test')
    >>> simple.register_simple_components(project)
    >>>
    >>> # Or with imbed (if installed):
    >>> # from ef.plugins import imbed
    >>> # imbed.register(project)
"""

# Import plugin modules
from ef.plugins import simple_plugin as simple
from ef.plugins import imbed_plugin as imbed

__all__ = ['simple', 'imbed']
```

## plugins/imbed_plugin.py

```python
"""
Plugin to integrate imbed's production implementations into ef.

This is a stub/bridge module that will connect ef to the imbed package
when it's available.

Usage:
    from ef import Project
    from ef.plugins import imbed

    project = Project.create('production')
    imbed.register(project)

    # Now has all imbed components
    print(project.embedders.keys())  # openai-small, openai-large, ...
"""


def register(project, *, include_datasets=True, include_utils=True):
    """
    Register all imbed components with an ef project.

    Args:
        project: An ef.Project instance
        include_datasets: Whether to include dataset-related components
        include_utils: Whether to add utility methods

    Raises:
        ImportError: If imbed package is not installed

    Example:
        >>> from ef import Project
        >>> from ef.plugins import imbed
        >>> project = Project.create('production')
        >>> imbed.register(project)  # doctest: +SKIP
    """
    try:
        import imbed
    except ImportError:
        raise ImportError(
            "The imbed package is required for this plugin.\n"
            "Install it with: pip install imbed\n"
            "Or install ef with imbed support: pip install ef[imbed]"
        )

    _register_embedders(project)
    _register_planarizers(project)
    _register_clusterers(project)
    _register_segmenters(project)

    if include_datasets:
        _register_dataset_classes(project)

    if include_utils:
        _add_utility_methods(project)


def _register_embedders(project):
    """Add imbed's real embedders to project."""
    try:
        from imbed.components.vectorization import embedders as imbed_embedders

        # Wrap each imbed embedder
        for name, func in imbed_embedders.items():
            project.embedders.register(name)(func)

        print(f"✓ Registered {len(imbed_embedders)} imbed embedders")
    except ImportError as e:
        print(f"○ Could not register imbed embedders: {e}")


def _register_planarizers(project):
    """Add imbed's real planarizers."""
    try:
        from imbed.components.planarization import planarizers as imbed_planarizers

        for name, func in imbed_planarizers.items():
            project.planarizers.register(name)(func)

        print(f"✓ Registered {len(imbed_planarizers)} imbed planarizers")
    except ImportError as e:
        print(f"○ Could not register imbed planarizers: {e}")


def _register_clusterers(project):
    """Add imbed's real clusterers."""
    try:
        from imbed.components.clusterization import clusterers as imbed_clusterers

        for name, func in imbed_clusterers.items():
            project.clusterers.register(name)(func)

        print(f"✓ Registered {len(imbed_clusterers)} imbed clusterers")
    except ImportError as e:
        print(f"○ Could not register imbed clusterers: {e}")


def _register_segmenters(project):
    """Add imbed's real segmenters."""
    try:
        from imbed.components.segmentation import segmenters as imbed_segmenters

        for name, func in imbed_segmenters.items():
            project.segmenters.register(name)(func)

        print(f"✓ Registered {len(imbed_segmenters)} imbed segmenters")
    except ImportError as e:
        print(f"○ Could not register imbed segmenters: {e}")


def _register_dataset_classes(project):
    """Add imbed's dataset classes."""
    # TODO: Implement when imbed dataset structure is stable
    pass


def _add_utility_methods(project):
    """Add imbed utility methods to project."""
    # TODO: Implement when imbed utility structure is stable
    pass


# Convenience functions


def register_all(project):
    """
    Convenience: register everything from imbed.

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('full')
    >>> imbed.register_all(project)  # doctest: +SKIP
    """
    register(project, include_datasets=True, include_utils=True)


def register_embedders_only(project):
    """
    Register only imbed embedders (no other components).

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('embed_only')
    >>> imbed.register_embedders_only(project)  # doctest: +SKIP
    """
    _register_embedders(project)


def register_ml_only(project):
    """
    Just embedders, planarizers, clusterers - no datasets.

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('ml_only')
    >>> imbed.register_ml_only(project)  # doctest: +SKIP
    """
    register(project, include_datasets=False, include_utils=False)
```

## plugins/simple_plugin.py

```python
"""
Simple built-in components for ef.

These toy implementations allow ef to work out-of-the-box without
requiring heavy ML dependencies.
"""

from collections.abc import Mapping
from typing import Any


def register_simple_components(project):
    """
    Register all simple (toy) components with a project.

    Args:
        project: An ef.Project instance

    Example:
        >>> from ef import Project
        >>> from ef.plugins import simple
        >>> project = Project.create('test', backend='memory')
        >>> simple.register_simple_components(project)
        >>> 'simple' in project.embedders
        True
    """
    _register_segmenters(project)
    _register_embedders(project)
    _register_planarizers(project)
    _register_clusterers(project)


def _register_segmenters(project):
    """Register simple segmentation components."""

    @project.segmenters.register('identity')
    def identity_segmenter(source: Any) -> dict[str, str]:
        """Return source as-is (no segmentation)."""
        if isinstance(source, str):
            return {'main': source}
        return source

    @project.segmenters.register('lines')
    def line_segmenter(source: str) -> dict[str, str]:
        """Split text into lines."""
        lines = source.split('\n')
        return {f'line_{i}': line for i, line in enumerate(lines) if line.strip()}

    @project.segmenters.register('sentences')
    def sentence_segmenter(source: str) -> dict[str, str]:
        """Split text into sentences (simple period-based)."""
        import re

        sentences = re.split(r'[.!?]+', source)
        return {
            f'sent_{i}': sent.strip()
            for i, sent in enumerate(sentences)
            if sent.strip()
        }


def _register_embedders(project):
    """Register simple embedding components."""

    @project.embedders.register('simple', dimension=3)
    def simple_embedder(segments: Mapping[str, str]) -> dict[str, list[float]]:
        """
        Simple embedder for testing.

        Counts characters, words, and punctuation.
        """
        result = {}
        for key, text in segments.items():
            n_chars = len(text)
            n_words = len(text.split())
            n_punct = sum(1 for c in text if c in '.,!?;:')
            result[key] = [float(n_chars), float(n_words), float(n_punct)]
        return result

    @project.embedders.register('char_counts', dimension=26)
    def char_count_embedder(segments: Mapping[str, str]) -> dict[str, list[float]]:
        """
        Embed text as character frequency vector.

        Returns a 26-dimensional vector of letter frequencies.
        """
        result = {}
        for key, text in segments.items():
            text_lower = text.lower()
            counts = [float(text_lower.count(chr(ord('a') + i))) for i in range(26)]
            result[key] = counts
        return result


def _register_planarizers(project):
    """Register simple planarization components."""

    @project.planarizers.register('simple_2d')
    def simple_planarizer(
        embeddings: Mapping[str, list[float]],
    ) -> dict[str, tuple[float, float]]:
        """
        Simple 2D projection (just takes first two dimensions).
        """
        return {
            key: (vec[0] if len(vec) > 0 else 0.0, vec[1] if len(vec) > 1 else 0.0)
            for key, vec in embeddings.items()
        }

    @project.planarizers.register('normalize_2d')
    def normalize_2d_planarizer(
        embeddings: Mapping[str, list[float]],
    ) -> dict[str, tuple[float, float]]:
        """
        Project to 2D and normalize to unit circle.
        """
        import math

        result = {}
        for key, vec in embeddings.items():
            x = vec[0] if len(vec) > 0 else 0.0
            y = vec[1] if len(vec) > 1 else 0.0

            # Normalize
            magnitude = math.sqrt(x * x + y * y)
            if magnitude > 0:
                x, y = x / magnitude, y / magnitude

            result[key] = (x, y)

        return result


def _register_clusterers(project):
    """Register simple clustering components."""

    @project.clusterers.register('simple_kmeans')
    def simple_clusterer(
        embeddings: Mapping[str, list[float]], *, n_clusters: int = 3
    ) -> dict[str, int]:
        """
        Simple clustering based on first dimension.

        Sorts by first dimension and splits into n groups.
        """
        import numpy as np

        keys = list(embeddings.keys())
        vecs = list(embeddings.values())

        if not vecs:
            return {}

        # Sort by first dimension and split into groups
        first_dims = [v[0] if v else 0.0 for v in vecs]
        sorted_indices = np.argsort(first_dims)

        clusters = {}
        items_per_cluster = len(keys) // n_clusters + 1

        for i, idx in enumerate(sorted_indices):
            cluster_id = i // items_per_cluster
            clusters[keys[idx]] = min(cluster_id, n_clusters - 1)

        return clusters

    @project.clusterers.register('threshold')
    def threshold_clusterer(
        embeddings: Mapping[str, list[float]], *, threshold: float = 10.0
    ) -> dict[str, int]:
        """
        Binary clustering based on magnitude threshold.

        Cluster 0: magnitude < threshold
        Cluster 1: magnitude >= threshold
        """
        import math

        result = {}
        for key, vec in embeddings.items():
            magnitude = math.sqrt(sum(x * x for x in vec))
            result[key] = 0 if magnitude < threshold else 1

        return result
```

## projects.py

```python
"""
Project management for ef (Embedding Flow).

This module provides the main Project and Projects classes for managing
embedding pipelines.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any
from types import SimpleNamespace
from collections.abc import MutableMapping, Iterator
import os
import tempfile

from ef.base import (
    ComponentRegistry,
    SegmentKey,
    Segment,
    Vector,
    PlanarVector,
    ClusterIndex,
)
from ef.storage import mk_project_mall
from ef.dag import assemble_pipeline_dag, DAG


# ============================================================================
# Default Component Registries
# ============================================================================


def _mk_default_registries() -> SimpleNamespace:
    """
    Create default component registries with simple (toy) implementations.

    Returns a SimpleNamespace with registries for:
    - segmenters
    - embedders
    - planarizers
    - clusterers

    These registries come pre-loaded with simple implementations so ef
    works out-of-the-box without requiring heavy dependencies.
    """
    registries = SimpleNamespace(
        segmenters=ComponentRegistry('segmenters'),
        embedders=ComponentRegistry('embedders'),
        planarizers=ComponentRegistry('planarizers'),
        clusterers=ComponentRegistry('clusterers'),
    )

    # Auto-register simple components from plugin
    from ef.plugins import simple

    simple.register_simple_components(
        # Create minimal project-like object for registration
        type(
            '_Project',
            (),
            {
                'segmenters': registries.segmenters,
                'embedders': registries.embedders,
                'planarizers': registries.planarizers,
                'clusterers': registries.clusterers,
            },
        )()
    )

    return registries


# ============================================================================
# Project Class
# ============================================================================


@dataclass
class Project:
    """
    Main project interface for ef pipelines.

    Provides:
    - Component registries (as Mapping stores)
    - Data storage (via mall - store of stores)
    - Pipeline assembly (via DAG composition)
    - Automatic persistence

    Example:
        >>> from ef import Project
        >>> project = Project.create('my_project', backend='memory')
        >>> project.add_source('doc1', 'Sample text')
        >>> _ = project.create_pipeline('test', embedder='simple')
        >>> results = project.run_pipeline('test')
        >>> 'embeddings' in results
        True
    """

    project_id: str
    mall: dict[str, MutableMapping] = field(default_factory=dict)
    registries: SimpleNamespace = field(default_factory=_mk_default_registries)
    pipelines: dict[str, DAG] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_id: str,
        *,
        root_dir: str | None = None,
        backend: str = 'files',
        registries: SimpleNamespace | None = None,
    ) -> 'Project':
        """
        Create a new project with storage and component registries.

        Args:
            project_id: Unique project identifier
            root_dir: Storage root directory
            backend: Storage backend ('files' or 'memory')
            registries: Custom registries (uses defaults if None)

        Returns:
            New Project instance

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> list(project.mall.keys())
            ['segments', 'embeddings', 'planar_embeddings', 'clusters']
        """
        mall = mk_project_mall(project_id, root_dir, backend=backend)

        if registries is None:
            registries = _mk_default_registries()

        return cls(
            project_id=project_id,
            mall=mall,
            registries=registries,
        )

    # -------- Component Access --------

    @property
    def segmenters(self) -> ComponentRegistry:
        """Access to segmentation components."""
        return self.registries.segmenters

    @property
    def embedders(self) -> ComponentRegistry:
        """Access to embedding components."""
        return self.registries.embedders

    @property
    def planarizers(self) -> ComponentRegistry:
        """Access to planarization components."""
        return self.registries.planarizers

    @property
    def clusterers(self) -> ComponentRegistry:
        """Access to clustering components."""
        return self.registries.clusterers

    # -------- Data Access --------

    @property
    def segments(self) -> MutableMapping[SegmentKey, Segment]:
        """Access to segments store."""
        return self.mall['segments']

    @property
    def embeddings(self) -> MutableMapping[SegmentKey, Vector]:
        """Access to embeddings store."""
        return self.mall['embeddings']

    @property
    def planar_embeddings(self) -> MutableMapping[SegmentKey, PlanarVector]:
        """Access to planar embeddings store."""
        return self.mall['planar_embeddings']

    @property
    def clusters(self) -> MutableMapping[SegmentKey, ClusterIndex]:
        """Access to clusters store."""
        return self.mall['clusters']

    # -------- Data Operations --------

    def add_source(self, key: str, source_data: Any) -> None:
        """
        Add source data to the project.

        The source will be stored as-is. You can later run a pipeline
        to segment and process it.

        Args:
            key: Unique identifier for this source
            source_data: The data to store (will be converted to string)

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> project.add_source('doc1', 'Sample text')
            >>> 'doc1' in project.segments
            True
        """
        self.segments[key] = str(source_data)

    def list_components(self, component_type: str | None = None) -> dict:
        """
        List available components, optionally filtered by type.

        Args:
            component_type: Type to filter ('segmenters', 'embedders', etc.)
                          If None, returns all components

        Returns:
            Dictionary of component types to lists of component names

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> components = project.list_components()
            >>> 'embedders' in components
            True
            >>> 'simple' in components['embedders']
            True
        """
        all_components = {
            'segmenters': list(self.segmenters.keys()),
            'embedders': list(self.embedders.keys()),
            'planarizers': list(self.planarizers.keys()),
            'clusterers': list(self.clusterers.keys()),
        }

        if component_type:
            return {component_type: all_components.get(component_type, [])}

        return all_components

    # -------- Pipeline Management --------

    def create_pipeline(
        self,
        name: str,
        *,
        segmenter: str | None = None,
        embedder: str | None = None,
        planarizer: str | None = None,
        clusterer: str | None = None,
        **component_params,
    ) -> DAG:
        """
        Create a pipeline from component names.

        Args:
            name: Pipeline name
            segmenter: Name of segmenter component (or None to skip)
            embedder: Name of embedder component (or None to skip)
            planarizer: Name of planarizer component (or None to skip)
            clusterer: Name of clusterer component (or None to skip)
            **component_params: Parameters to pass to components (e.g., n_clusters=5)

        Returns:
            The assembled DAG pipeline

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('my_pipeline', embedder='simple')
            >>> 'my_pipeline' in project.pipelines
            True
        """
        # Get component functions
        seg_func = self.segmenters[segmenter] if segmenter else None
        emb_func = self.embedders[embedder] if embedder else None
        pla_func = self.planarizers[planarizer] if planarizer else None
        clu_func = self.clusterers[clusterer] if clusterer else None

        # Apply any parameters via partial
        if component_params:
            if clu_func and 'n_clusters' in component_params:
                clu_func = partial(clu_func, n_clusters=component_params['n_clusters'])

        # Assemble DAG
        dag = assemble_pipeline_dag(
            segmenter=seg_func,
            embedder=emb_func,
            planarizer=pla_func,
            clusterer=clu_func,
        )

        # Store pipeline
        self.pipelines[name] = dag

        return dag

    def run_pipeline(
        self,
        pipeline_name: str,
        *,
        source_key: str | None = None,
        source_data: Any | None = None,
        persist: bool = True,
    ) -> dict:
        """
        Run a pipeline on source data.

        Args:
            pipeline_name: Name of the pipeline to run
            source_key: Key of source data in segments store (if already added)
            source_data: Source data to process (if not already in store)
            persist: Whether to persist intermediate results to stores

        Returns:
            Dictionary with pipeline results

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('simple', embedder='simple')
            >>> project.segments['doc1'] = 'Test document'
            >>> results = project.run_pipeline('simple', source_key='doc1')
            >>> 'embeddings' in results
            True
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        dag = self.pipelines[pipeline_name]

        # Prepare input - treat as segments for the pipeline
        if source_data is not None:
            # Use provided source data
            if isinstance(source_data, str):
                segments = {'main': source_data}
            else:
                segments = source_data
        elif source_key is not None:
            # Load from segments store
            segments = {source_key: self.segments[source_key]}
        else:
            # Use all segments
            segments = dict(self.segments)

        # Run the DAG with segments as input
        results = dag(segments=segments)

        # Persist results if requested
        if persist:
            if 'segments' in results:
                self.segments.update(results['segments'])
            if 'embeddings' in results:
                self.embeddings.update(results['embeddings'])
            if 'planar_embeddings' in results:
                self.planar_embeddings.update(results['planar_embeddings'])
            if 'clusters' in results:
                self.clusters.update(results['clusters'])

        return results

    def list_pipelines(self) -> list[str]:
        """
        List all created pipelines.

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('pipe1', embedder='simple')
            >>> 'pipe1' in project.list_pipelines()
            True
        """
        return list(self.pipelines.keys())

    # -------- Convenience Methods --------

    def quick_embed(
        self,
        source: str | dict[str, str],
        *,
        embedder: str = 'simple',
        segmenter: str | None = None,
    ) -> dict[str, Vector]:
        """
        Quick embedding without creating a named pipeline.

        Args:
            source: Text or dict of texts to embed
            embedder: Name of embedder to use
            segmenter: Optional segmenter name

        Returns:
            Dictionary of embeddings

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> embeddings = project.quick_embed('Test text', embedder='simple')
            >>> 'main' in embeddings
            True
        """
        # Create temporary pipeline
        temp_name = f'_temp_{id(self)}'
        self.create_pipeline(
            temp_name,
            segmenter=segmenter,
            embedder=embedder,
        )

        # Run and cleanup
        try:
            results = self.run_pipeline(
                temp_name,
                source_data=source,
                persist=False,
            )
            return results.get('embeddings', {})
        finally:
            del self.pipelines[temp_name]

    def summary(self) -> dict:
        """
        Get a summary of the project state.

        Returns:
            Dictionary with project statistics

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> summary = project.summary()
            >>> 'project_id' in summary
            True
        """
        return {
            'project_id': self.project_id,
            'n_segments': len(self.segments),
            'n_embeddings': len(self.embeddings),
            'n_planar_embeddings': len(self.planar_embeddings),
            'n_clusters': len(self.clusters),
            'n_pipelines': len(self.pipelines),
            'available_components': self.list_components(),
        }


# ============================================================================
# Projects Manager
# ============================================================================


class Projects(MutableMapping[str, Project]):
    """
    A store of projects, following the "mall" pattern.

    This allows you to manage multiple projects with a dict-like interface.

    Example:
        >>> from ef import Projects
        >>> projects = Projects()
        >>> proj = projects.create_project('proj1', backend='memory')
        >>> 'proj1' in projects
        True
    """

    def __init__(self, root_dir: str | None = None):
        """
        Initialize projects manager.

        Args:
            root_dir: Root directory for all projects
        """
        self.root_dir = root_dir or os.path.join(tempfile.gettempdir(), 'ef_projects')
        self._projects: dict[str, Project] = {}

    def __getitem__(self, key: str) -> Project:
        if key not in self._projects:
            # Try to load from disk
            project_dir = os.path.join(self.root_dir, key)
            if os.path.exists(project_dir):
                self._projects[key] = Project.create(
                    key, root_dir=self.root_dir, backend='files'
                )
            else:
                raise KeyError(f"Project '{key}' not found")
        return self._projects[key]

    def __setitem__(self, key: str, value: Project) -> None:
        if not isinstance(value, Project):
            raise TypeError(f"Value must be a Project, got {type(value)}")
        self._projects[key] = value

    def __delitem__(self, key: str) -> None:
        del self._projects[key]
        # Optionally delete from disk
        import shutil

        project_dir = os.path.join(self.root_dir, key)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

    def __iter__(self) -> Iterator[str]:
        # List all projects in memory and on disk
        disk_projects = set()
        if os.path.exists(self.root_dir):
            disk_projects = {
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            }
        return iter(set(self._projects.keys()) | disk_projects)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def create_project(self, project_id: str, **kwargs) -> Project:
        """
        Create a new project and add it to the manager.

        Args:
            project_id: Unique project identifier
            **kwargs: Additional arguments for Project.create()

        Returns:
            New Project instance

        Example:
            >>> projects = Projects()
            >>> proj = projects.create_project('new_proj', backend='memory')
            >>> proj.project_id
            'new_proj'
        """
        project = Project.create(project_id, root_dir=self.root_dir, **kwargs)
        self[project_id] = project
        return project
```

## storage.py

```python
"""
Storage layer for ef using the Mall pattern (store of stores).

This module provides:
- Storage backends with MutableMapping interfaces
- Extension-based file stores
- Mall creation for project data
- Optional dol integration with fallbacks
"""

import os
import tempfile
from collections.abc import MutableMapping, Iterator
from typing import Any

# Optional dol import with fallback
try:
    from dol import Files, wrap_kvs, add_ipython_key_completions

    HAVE_DOL = True
except ImportError:
    HAVE_DOL = False

    def add_ipython_key_completions(obj):
        """Fallback: no-op decorator."""
        return obj


# ============================================================================
# File Storage Backends
# ============================================================================


class SimpleFileStore(MutableMapping):
    """
    Simple file-based storage (fallback when dol not available).

    Provides MutableMapping interface to filesystem storage.
    """

    def __init__(self, rootdir: str, extension: str = 'pkl'):
        self.rootdir = rootdir
        self.extension = extension
        os.makedirs(rootdir, exist_ok=True)

    def _filepath(self, key: str) -> str:
        return os.path.join(self.rootdir, f"{key}.{self.extension}")

    def __getitem__(self, key: str) -> Any:
        filepath = self._filepath(key)

        import pickle
        import json

        with open(filepath, 'rb') as f:
            data = f.read()

        if self.extension == 'pkl':
            return pickle.loads(data)
        elif self.extension == 'json':
            return json.loads(data.decode())
        else:
            return data.decode()

    def __setitem__(self, key: str, value: Any) -> None:
        filepath = self._filepath(key)

        import pickle
        import json

        if self.extension == 'pkl':
            data = pickle.dumps(value)
        elif self.extension == 'json':
            data = json.dumps(value).encode()
        else:
            data = str(value).encode()

        with open(filepath, 'wb') as f:
            f.write(data)

    def __delitem__(self, key: str) -> None:
        os.remove(self._filepath(key))

    def __iter__(self) -> Iterator[str]:
        for filename in os.listdir(self.rootdir):
            if filename.endswith(f'.{self.extension}'):
                yield filename.rsplit('.', 1)[0]

    def __len__(self) -> int:
        return sum(1 for _ in self)


def mk_extension_based_store(rootdir: str, *, extension: str = 'pkl') -> MutableMapping:
    """
    Create a storage backend with extension-based serialization.

    Uses dol if available, otherwise uses SimpleFileStore fallback.

    Args:
        rootdir: Base directory for storage
        extension: File extension ('pkl', 'json', 'txt')

    Returns:
        MutableMapping interface to storage

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = mk_extension_based_store(tmpdir, extension='json')
    ...     store['test'] = {'key': 'value'}
    ...     assert store['test'] == {'key': 'value'}
    """
    import pickle
    import json

    os.makedirs(rootdir, exist_ok=True)

    if HAVE_DOL:
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

        return add_ipython_key_completions(store)
    else:
        # Use simple fallback
        return SimpleFileStore(rootdir, extension)


# ============================================================================
# Mall Pattern (Store of Stores)
# ============================================================================


def mk_project_mall(
    project_id: str,
    root_dir: str | None = None,
    *,
    backend: str = 'files',
) -> dict[str, MutableMapping]:
    """
    Create a "mall" (store of stores) for a project.

    A mall provides separate storage for each pipeline stage:
    - segments: text segments
    - embeddings: vector embeddings
    - planar_embeddings: 2D coordinates
    - clusters: cluster assignments

    Args:
        project_id: Unique identifier for the project
        root_dir: Base directory for storage (uses temp if None)
        backend: Storage backend ('files', 'memory')

    Returns:
        Dictionary mapping stage names to storage objects

    >>> mall = mk_project_mall('test_project', backend='memory')
    >>> list(mall.keys())
    ['segments', 'embeddings', 'planar_embeddings', 'clusters']
    """
    if root_dir is None:
        root_dir = os.path.join(tempfile.gettempdir(), 'ef_projects', project_id)

    if backend == 'memory':
        # Use simple dicts for in-memory storage
        return {
            'segments': {},
            'embeddings': {},
            'planar_embeddings': {},
            'clusters': {},
        }
    elif backend == 'files':
        # Use filesystem storage with appropriate serialization
        return {
            'segments': mk_extension_based_store(
                os.path.join(root_dir, 'segments'), extension='txt'
            ),
            'embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'embeddings'), extension='pkl'
            ),
            'planar_embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'planar_embeddings'), extension='json'
            ),
            'clusters': mk_extension_based_store(
                os.path.join(root_dir, 'clusters'), extension='json'
            ),
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

## tests/__init__.py

```python
"""Tests for ef package."""
```

## tests/test_basic.py

```python
"""Basic tests for ef package."""

import pytest
from ef import Project, Projects, ComponentRegistry


def test_project_creation():
    """Test basic project creation."""
    project = Project.create('test', backend='memory')
    assert project.project_id == 'test'
    assert len(project.segments) == 0


def test_add_source():
    """Test adding source data."""
    project = Project.create('test', backend='memory')
    project.add_source('doc1', 'Sample text')

    assert 'doc1' in project.segments
    assert project.segments['doc1'] == 'Sample text'


def test_component_discovery():
    """Test listing components."""
    project = Project.create('test', backend='memory')
    components = project.list_components()

    assert 'embedders' in components
    assert 'clusterers' in components
    assert 'simple' in components['embedders']


def test_pipeline_creation():
    """Test creating a pipeline."""
    project = Project.create('test', backend='memory')
    project.create_pipeline('test_pipe', embedder='simple')

    assert 'test_pipe' in project.pipelines
    assert 'test_pipe' in project.list_pipelines()


def test_pipeline_execution():
    """Test running a pipeline."""
    project = Project.create('test', backend='memory')
    project.add_source('doc1', 'Test document')

    project.create_pipeline('test', embedder='simple')
    results = project.run_pipeline('test')

    assert 'embeddings' in results
    assert 'doc1' in results['embeddings']
    assert len(results['embeddings']['doc1']) == 3  # simple embedder returns 3D


def test_quick_embed():
    """Test quick embed functionality."""
    project = Project.create('test', backend='memory')
    embeddings = project.quick_embed('Test text')

    assert 'main' in embeddings
    assert len(embeddings['main']) == 3


def test_custom_component():
    """Test registering custom component."""
    project = Project.create('test', backend='memory')

    @project.embedders.register('custom', dimension=2)
    def custom_embedder(segments):
        return {k: [1.0, 2.0] for k in segments}

    assert 'custom' in project.embedders
    meta = project.embedders.get_metadata('custom')
    assert meta['dimension'] == 2


def test_component_registry():
    """Test ComponentRegistry."""
    registry = ComponentRegistry('test')

    # Add component
    registry['func1'] = lambda x: x * 2

    # Test access
    assert 'func1' in registry
    assert len(registry) == 1
    assert list(registry.keys()) == ['func1']

    # Test decorator
    @registry.register('func2', param=42)
    def func2(x):
        return x + 1

    assert 'func2' in registry
    assert registry.get_metadata('func2')['param'] == 42


def test_projects_manager():
    """Test Projects manager."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        projects = Projects(root_dir=tmpdir)

        # Create project
        proj1 = projects.create_project('proj1', backend='memory')
        assert proj1.project_id == 'proj1'

        # Access project
        assert 'proj1' in projects
        retrieved = projects['proj1']
        assert retrieved.project_id == 'proj1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## README.md

```python
# ef (Embedding Flow)

**Lightweight framework for embedding pipelines**

ef is a simple, composable framework for building and running embedding pipelines. It provides:
- ✅ Works out-of-the-box (zero configuration, built-in components)
- ✅ Component registries as mapping stores (easy discovery)
- ✅ Automatic pipeline composition (via DAG)
- ✅ Flexible storage backends (memory, files, custom)
- ✅ Plugin system (add production components when needed)

## Installation

```bash
# Basic installation (works immediately with built-in components)
pip install ef

# With full functionality (dol, meshed, larder)
pip install ef[full]

# With imbed integration (production components)
pip install ef[imbed]
```

## Quick Start

```python
from ef import Project

# Create project (works immediately!)
project = Project.create('my_project', backend='memory')

# Add data
project.add_source('doc1', 'First document about AI')
project.add_source('doc2', 'Second document about ML')

# List available components
print(project.list_components())
# {
#   'embedders': ['simple', 'char_counts'],
#   'planarizers': ['simple_2d', 'normalize_2d'],
#   'clusterers': ['simple_kmeans', 'threshold'],
#   'segmenters': ['identity', 'lines', 'sentences']
# }

# Create pipeline
project.create_pipeline(
    'analysis',
    embedder='simple',
    planarizer='simple_2d',
    clusterer='simple_kmeans',
    n_clusters=2
)

# Run pipeline (persists all results automatically)
results = project.run_pipeline('analysis')

# Access persisted data
print(f"Segments: {len(project.segments)}")
print(f"Embeddings: {len(project.embeddings)}")
print(f"Clusters: {dict(project.clusters)}")

# Get project summary
print(project.summary())
```

## Core Concepts

### 1. Component Registries (Mapping Stores)

Components are stored in registries that behave like dictionaries:

```python
# Access components like a dict
embedder = project.embedders['simple']
vectors = embedder({'text1': 'Sample text'})

# List all components
print(list(project.embedders.keys()))

# Get component metadata
meta = project.embedders.get_metadata('simple')
```

### 2. Mall Pattern (Store of Stores)

Each project has a "mall" - separate stores for each data type:

```python
# Access different stores
project.segments['doc1'] = 'text'
project.embeddings['doc1'] = [1.0, 2.0, 3.0]
project.clusters['doc1'] = 0

# All stores use MutableMapping interface
for key, value in project.embeddings.items():
    print(f"{key}: {value}")
```

### 3. Pipeline Assembly

Pipelines are assembled automatically from components:

```python
# Create pipeline by naming components
project.create_pipeline(
    'my_pipeline',
    segmenter='lines',      # Optional: split text
    embedder='simple',      # Required: embed segments
    planarizer='simple_2d', # Optional: reduce dimensions
    clusterer='simple_kmeans',  # Optional: cluster
    n_clusters=5  # Pass parameters to components
)

# Run with automatic persistence
results = project.run_pipeline('my_pipeline')
```

### 4. Flexible Storage

Choose storage backend based on needs:

```python
# In-memory (fast, temporary)
project = Project.create('test', backend='memory')

# File-based (persistent)
project = Project.create('prod', backend='files', root_dir='/data')

# Custom (bring your own store)
from ef.storage import mk_project_mall
mall = mk_project_mall('custom', backend='files')
mall['embeddings'] = MyCustomStore()  # Any MutableMapping
project = Project('custom', mall=mall)
```

## Plugin System

### Built-in Components (Always Available)

ef comes with simple implementations that work without dependencies:

```python
# Automatically registered on import
from ef import Project
project = Project.create('test')

# Has built-in components:
# - Embedders: simple, char_counts
# - Planarizers: simple_2d, normalize_2d
# - Clusterers: simple_kmeans, threshold
# - Segmenters: identity, lines, sentences
```

### Adding Production Components

Use plugins to add real ML implementations:

```python
from ef import Project
from ef.plugins import imbed  # Requires: pip install ef[imbed]

project = Project.create('production')
imbed.register(project)

# Now has production components:
# - OpenAI embedders
# - UMAP planarization
# - scikit-learn clustering
# - And more...

print(list(project.embedders.keys()))
# ['simple', 'char_counts', 'openai-small', 'openai-large', ...]
```

### Writing Your Own Plugin

```python
# my_plugin.py
def register(project):
    """Add custom components to project."""
    
    @project.embedders.register('my_embedder', dimension=768)
    def my_embedder(segments):
        # Your implementation
        return {key: compute(text) for key, text in segments.items()}

# Use it
from ef import Project
import my_plugin

project = Project.create('custom')
my_plugin.register(project)
```

## Advanced Usage

### Multiple Projects

```python
from ef import Projects

# Manage multiple projects
projects = Projects(root_dir='/data')

# Create projects
proj1 = projects.create_project('research', backend='files')
proj2 = projects.create_project('production', backend='files')

# Access later
existing = projects['research']

# List all
print(list(projects.keys()))
```

### Quick Embed (No Pipeline)

```python
# For one-off embeddings
embeddings = project.quick_embed(
    'Some text to embed',
    embedder='simple'
)
```

### Custom Components

```python
# Register your own component
@project.embedders.register('custom', dimension=512)
def custom_embedder(segments):
    return {k: my_model(v) for k, v in segments.items()}

# Use in pipeline
project.create_pipeline('custom_pipe', embedder='custom')
```

## Architecture

ef follows Option 1 from the design plan:

```
┌─────────────────────────────────────┐
│  ef (lightweight interface layer)   │
│  - ComponentRegistry                │
│  - Project/Projects                 │
│  - Mall pattern                     │
│  - Pipeline assembly                │
│  - Built-in toy components          │
└──────────────┬──────────────────────┘
               │ imports (optional)
               ↓
┌─────────────────────────────────────┐
│  imbed (heavy implementation)       │
│  - Real embedders (OpenAI, etc.)    │
│  - Real planarizers (UMAP)          │
│  - Real clusterers (sklearn)        │
│  - Dataset classes                  │
│  - All utilities                    │
└─────────────────────────────────────┘
```

## Design Principles

1. **Works immediately**: Built-in components require no setup
2. **Mapping everywhere**: All stores use `MutableMapping` interface
3. **Composable**: Mix and match components easily
4. **Discoverable**: `.list_components()`, `.list_pipelines()`
5. **Flexible**: Swap storage backends without code changes
6. **Extensible**: Plugin system for adding functionality
7. **Progressive enhancement**: Start simple, add complexity as needed

## Dependencies

**Required (minimal):**
- Python 3.10+
- numpy

**Optional (recommended):**
- `dol>=0.2.38` - Better storage abstraction
- `meshed>=0.1.20` - Automatic DAG composition
- `larder>=0.1.6` - Automatic caching

**Plugin dependencies:**
- `imbed>=0.1` - Production ML components

Install optional dependencies:
```bash
pip install ef[full]     # Install dol, meshed, larder
pip install ef[imbed]    # Install imbed + full dependencies
```

## Development

```bash
# Clone repository
git clone https://github.com/thorwhalen/ef.git
cd ef

# Install in development mode
pip install -e .

# Run tests (if available)
pytest
```

## Examples

See the `imbed_refactored/` directory for detailed examples:
- `imbed_refactored.py` - Core patterns and complete demo
- `advanced_example.py` - Real ML integrations (OpenAI, UMAP, sklearn)
- `persistence_examples.py` - Pipeline sharing and caching

## Comparison with imbed

| Feature | ef | imbed |
|---------|----|----|
| **Purpose** | Lightweight interface framework | Production ML implementations |
| **Dependencies** | numpy (+ optional) | openai, umap, sklearn, datasets, etc. |
| **Out-of-box** | ✓ Works immediately | Requires configuration |
| **Components** | Toy implementations | Production implementations |
| **Use case** | Prototyping, learning, interfaces | Production ML pipelines |

**Use together:**
```python
from ef import Project
from ef.plugins import imbed

project = Project.create('best_of_both')
imbed.register(project)  # Add production power to clean interface
```

## License

MIT

## Contributing

Contributions welcome! Please:
1. Write tests for new features
2. Follow existing code style
3. Update documentation
4. Submit PRs to main branch

## Links

- **GitHub**: https://github.com/thorwhalen/ef
- **imbed**: https://github.com/thorwhalen/imbed (production components)
- **dol**: https://github.com/i2mint/dol (storage layer)
- **meshed**: https://github.com/i2mint/meshed (DAG composition) -- Tools for workflows involving semantic embeddings
```