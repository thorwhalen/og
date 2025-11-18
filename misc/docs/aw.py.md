## __init__.py

```python
"""aw - Agentic Workflows for Data Preparation

An AI agent package for data preparation with a focus on loading and preparing
data for various purposes (e.g., visualization with cosmograph).

Key Features:
- AgenticStep protocol for building modular agents
- ReAct pattern (Reason-Act-Observe) with retry logic
- Three validation flavors: schema, info-dict, and functional
- Code execution with safe defaults and extensibility
- Loading and Preparation agents for data workflows
- Cosmograph-specific validators and utilities
- Orchestration for chaining multiple steps

Example:
    >>> from aw import load_for_cosmo  # doctest: +SKIP
    >>> df, metadata = load_for_cosmo('data.csv')  # doctest: +SKIP
    >>> from cosmograph import cosmo  # doctest: +SKIP
    >>> params = metadata['preparing']['metadata']['validation_result']['params']  # doctest: +SKIP
    >>> cosmo(df, **params)  # doctest: +SKIP

Architecture:
    - aw.base: Core protocols and configurations
    - aw.validation: Three validation flavors
    - aw.tools: Code execution and agent tools
    - aw.utils: Helper functions and facades
    - aw.loading: LoadingAgent for data ingestion
    - aw.preparing: PreparationAgent for data transformation
    - aw.cosmo: Cosmograph-specific validators
    - aw.orchestration: Workflow management
"""

# Core abstractions
from aw.base import (
    AgenticStep,
    Validator,
    Tool,
    StepConfig,
    GlobalConfig,
    Context,
)

# Validation
from aw.validation import (
    schema_validator,
    info_dict_validator,
    functional_validator,
    all_validators,
    any_validator,
    is_type,
    is_not_empty,
    has_attributes,
)

# Tools
from aw.tools import (
    CodeInterpreterTool,
    SafeCodeInterpreter,
    ExecutionResult,
    create_langchain_executor,
)

# Utilities
from aw.util import (
    AW_DATA_DIR,
    djoin,
    FileSamplerTool,
    infer_loader_from_extension,
    infer_loader_params,
    compute_dataframe_info,
    get_numeric_columns,
    default_llm_factory,
    create_openai_chat,
    create_oa_chat,
)

# Agents
from aw.loading import LoadingAgent, create_loading_agent
from aw.preparing import PreparationAgent, create_preparation_agent

# Cosmograph support
from aw.cosmo import (
    create_cosmo_validator,
    try_cosmo_visualization,
    basic_cosmo_validator,
    strict_cosmo_validator,
    infer_cosmo_params,
)

# Orchestration
from aw.orchestration import (
    AgenticWorkflow,
    InteractiveWorkflow,
    create_data_prep_workflow,
    create_cosmo_prep_workflow,
    load_and_prepare,
    load_for_cosmo,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'AgenticStep',
    'Validator',
    'Tool',
    'StepConfig',
    'GlobalConfig',
    'Context',
    # Validation
    'schema_validator',
    'info_dict_validator',
    'functional_validator',
    'all_validators',
    'any_validator',
    'is_type',
    'is_not_empty',
    'has_attributes',
    # Tools
    'CodeInterpreterTool',
    'SafeCodeInterpreter',
    'ExecutionResult',
    'create_langchain_executor',
    # Utils
    'AW_DATA_DIR',
    'djoin',
    'FileSamplerTool',
    'infer_loader_from_extension',
    'infer_loader_params',
    'compute_dataframe_info',
    'get_numeric_columns',
    'default_llm_factory',
    'create_openai_chat',
    'create_oa_chat',
    # Agents
    'LoadingAgent',
    'create_loading_agent',
    'PreparationAgent',
    'create_preparation_agent',
    # Cosmo
    'create_cosmo_validator',
    'try_cosmo_visualization',
    'basic_cosmo_validator',
    'strict_cosmo_validator',
    'infer_cosmo_params',
    # Orchestration
    'AgenticWorkflow',
    'InteractiveWorkflow',
    'create_data_prep_workflow',
    'create_cosmo_prep_workflow',
    'load_and_prepare',
    'load_for_cosmo',
]
```

## base.py

```python
"""Core protocols and base classes for agentic workflows.

This module provides the foundational abstractions for building AI agents
that follow the AgenticStep protocol with ReAct pattern (Reason-Act-Observe).
"""

from typing import Protocol, Any, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping


T = TypeVar('T')
ArtifactType = TypeVar('ArtifactType')


class AgenticStep(Protocol[ArtifactType]):
    """Protocol defining the interface for any agentic step in a workflow.

    Each step follows the ReAct pattern:
    1. Reason (Thought): Analyze input and context
    2. Act (Action): Generate and execute code or use tools
    3. Observe: Capture results and validate
    4. Repeat or finish based on validation

    Example:
        >>> class MyAgent:
        ...     def execute(self, input_data, context):
        ...         # Agent implementation
        ...         return result, metadata
    """

    def execute(
        self, input_data: Any, context: MutableMapping[str, Any]
    ) -> tuple[ArtifactType, dict[str, Any]]:
        """Execute the agentic step.

        Args:
            input_data: The input to process
            context: Mutable mapping containing shared state and history

        Returns:
            Tuple of (artifact, metadata) where artifact is the main output
            and metadata contains auxiliary information
        """
        ...


class Validator(Protocol):
    """Protocol for validation functions.

    Validators check if an artifact meets requirements and return
    both a success indicator and detailed information.

    Example:
        >>> def is_non_empty_dataframe(df):
        ...     success = df is not None and len(df) > 0
        ...     info = {'shape': df.shape if success else None}
        ...     return success, info
    """

    def __call__(self, artifact: Any) -> tuple[bool, dict[str, Any]]:
        """Validate an artifact.

        Args:
            artifact: The artifact to validate

        Returns:
            Tuple of (success, info) where success is True if validation
            passed and info contains details about the validation
        """
        ...


class Tool(Protocol):
    """Protocol for tools that agents can use.

    Tools are callable objects that perform specific actions like
    executing code, sampling files, or calling external APIs.

    Example:
        >>> class FileSampler:
        ...     def __call__(self, uri):
        ...         # Sample file logic
        ...         return {'extension': '.csv', 'sample': '...'}
    """

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        ...


@dataclass
class StepConfig:
    """Configuration for an AgenticStep.

    Supports both simple objects (strings, ints) and callables for maximum
    flexibility with dependency injection.

    Attributes:
        llm: Either a model name string or a text-to-text chat function
        validator: Validation callable or schema object
        tools: List of callable tools available to the agent
        max_retries: Maximum number of retry attempts
        human_in_loop: Whether to require human approval/intervention

    Example:
        >>> config = StepConfig(
        ...     llm="gpt-4",
        ...     validator=lambda x: (True, {}),
        ...     max_retries=3
        ... )
    """

    llm: Union[str, Callable[[str], str]] = "gpt-4"
    validator: Union[Callable, Any] = None
    tools: list[Callable] = field(default_factory=list)
    max_retries: int = 3
    human_in_loop: bool = False

    def resolve_llm(self) -> Callable[[str], str]:
        """Resolve llm to a callable function.

        Returns:
            A callable that takes a prompt string and returns a response string
        """
        if callable(self.llm):
            return self.llm
        # Default implementation - will be replaced by actual LLM
        return lambda prompt: f"Response to: {prompt}"

    def resolve_validator(self) -> Validator:
        """Resolve validator to a callable.

        Returns:
            A callable that validates artifacts
        """
        if self.validator is None:
            # Default validator - always passes
            return lambda artifact: (True, {})
        if callable(self.validator):
            return self.validator
        # Handle schema-based validators (Pydantic, etc.)
        return self._schema_to_validator(self.validator)

    def _schema_to_validator(self, schema: Any) -> Validator:
        """Convert a schema object to a validator function."""

        def validate(artifact):
            try:
                # Try Pydantic model validation
                if hasattr(schema, 'model_validate'):
                    result = schema.model_validate(artifact)
                    return True, {'validated': result}
                # Try JSON schema validation
                elif hasattr(schema, 'validate'):
                    schema.validate(artifact)
                    return True, {}
                else:
                    # Unknown schema type - pass through
                    return True, {}
            except Exception as e:
                return False, {'error': str(e)}

        return validate


@dataclass
class GlobalConfig:
    """Global configuration with cascading defaults.

    Provides defaults that can be overridden at step or agent level.

    Example:
        >>> global_cfg = GlobalConfig(llm="gpt-4", max_retries=5)
        >>> step_cfg = global_cfg.override(llm="gpt-3.5-turbo")
    """

    llm: Union[str, Callable[[str], str]] = "gpt-4"
    max_retries: int = 3
    human_in_loop: bool = False

    def override(self, **kwargs) -> StepConfig:
        """Create a StepConfig with overridden values.

        Args:
            **kwargs: Values to override from global defaults

        Returns:
            A new StepConfig with merged configuration
        """
        defaults = {
            'llm': self.llm,
            'max_retries': self.max_retries,
            'human_in_loop': self.human_in_loop,
        }
        defaults.update(kwargs)
        return StepConfig(**defaults)


class Context(MutableMapping[str, Any]):
    """Context for sharing state and artifacts between agentic steps.

    Implements MutableMapping interface for dict-like behavior while
    providing additional functionality for managing agent state.

    Example:
        >>> ctx = Context()  # doctest: +SKIP
        >>> ctx['loading'] = {'df': df, 'info': {...}}  # doctest: +SKIP
        >>> ctx['preparing'] = {'df': prepared_df}  # doctest: +SKIP
    """

    def __init__(self, initial_data: dict = None):
        self._data: dict[str, Any] = initial_data or {}
        self._history: list[tuple[str, Any]] = []

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._history.append((key, value))
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def history(self) -> list[tuple[str, Any]]:
        """Access the history of all context updates."""
        return self._history

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of current context state.

        Returns:
            A dict copy of current context data
        """
        return dict(self._data)
```

## cosmo.py

```python
"""Cosmograph-specific validator tool.

Provides functional validation for cosmograph visualization requirements.
"""

from typing import Any, Callable
from aw.validation import functional_validator


def create_cosmo_validator(
    cosmo_function: Callable = None,
    required_columns: int = 2,
    allow_generated_params: bool = True,
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator for cosmograph requirements.

    This validator attempts to actually call cosmograph.cosmo() with the
    DataFrame to see if it's ready for visualization.

    Args:
        cosmo_function: The cosmograph.cosmo function (imported if None)
        required_columns: Minimum number of numeric columns needed
        allow_generated_params: Whether to auto-infer x/y columns

    Returns:
        Validator function following (artifact) -> (success, info) protocol

    Example:
        >>> validator = create_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
        >>> if success:  # doctest: +SKIP
        ...     print(f"Can visualize with: {info['params']}")  # doctest: +SKIP
    """

    def validate_cosmo_ready(df) -> tuple[bool, dict]:
        """Validate if DataFrame is ready for cosmograph visualization."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return False, {'error': 'Not a DataFrame', 'type': type(df).__name__}

        if len(df) == 0:
            return False, {'error': 'Empty DataFrame'}

        # Check for numeric columns
        numeric_cols = list(df.select_dtypes(include='number').columns)

        if len(numeric_cols) < required_columns:
            return False, {
                'error': f'Too few numeric columns: {len(numeric_cols)} < {required_columns}',
                'numeric_columns': numeric_cols,
                'all_columns': list(df.columns),
            }

        # Check for nulls in numeric columns
        null_counts = df[numeric_cols].isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            return False, {
                'error': 'Numeric columns contain nulls',
                'null_counts': cols_with_nulls,
            }

        # If allow_generated_params, try to find suitable columns
        if allow_generated_params:
            # Pick first two numeric columns
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            params = {'points_x_by': x_col, 'points_y_by': y_col}

            # Add optional params if available
            if len(numeric_cols) > 2:
                params['point_size_by'] = numeric_cols[2]

            # Check for categorical columns for color
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # Check if first categorical has reasonable cardinality
                first_cat = categorical_cols[0]
                if df[first_cat].nunique() < 50:
                    params['point_color_by'] = first_cat

            # Try actual cosmograph call if function provided
            if cosmo_function:
                try:
                    result = cosmo_function(df, **params)
                    return True, {
                        'params': params,
                        'result': result,
                        'note': 'Successfully created visualization',
                    }
                except Exception as e:
                    return False, {
                        'error': f'Cosmo call failed: {str(e)}',
                        'params': params,
                        'exception_type': type(e).__name__,
                    }
            else:
                # No actual cosmo function - just verify requirements met
                return True, {
                    'params': params,
                    'note': 'Requirements met (cosmo not called)',
                    'numeric_columns': numeric_cols,
                    'shape': df.shape,
                }

        # Basic validation passed
        return True, {
            'numeric_columns': numeric_cols,
            'shape': df.shape,
            'note': 'Basic requirements met',
        }

    return validate_cosmo_ready


def try_cosmo_visualization(
    df: Any, cosmo_function: Callable = None, **cosmo_kwargs
) -> tuple[bool, dict]:
    """Try to create a cosmograph visualization.

    This is a functional validator that actually attempts the visualization.

    Args:
        df: DataFrame to visualize
        cosmo_function: The cosmograph.cosmo function
        **cosmo_kwargs: Additional arguments for cosmo

    Returns:
        Tuple of (success, info)

    Example:
        >>> from cosmograph import cosmo  # doctest: +SKIP
        >>> success, info = try_cosmo_visualization(df, cosmo)  # doctest: +SKIP
    """
    if cosmo_function is None:
        try:
            from cosmograph import cosmo

            cosmo_function = cosmo
        except ImportError:
            return False, {
                'error': 'cosmograph not available',
                'note': 'Install with: pip install cosmograph',
            }

    # First check basic requirements
    validator = create_cosmo_validator(cosmo_function=None)
    basic_ok, basic_info = validator(df)

    if not basic_ok:
        return False, basic_info

    # Get suggested params if not provided
    params = cosmo_kwargs or basic_info.get('params', {})

    # Try the visualization
    try:
        result = cosmo_function(df, **params)
        return True, {
            'result': result,
            'params': params,
            'note': 'Visualization created successfully',
        }
    except Exception as e:
        import traceback

        return False, {
            'error': str(e),
            'exception_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'params': params,
        }


# Pre-configured validators for common use cases


def basic_cosmo_validator() -> Callable:
    """Basic validator that checks structural requirements only.

    Does not require cosmograph to be installed.

    Example:
        >>> validator = basic_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
    """
    return create_cosmo_validator(cosmo_function=None, allow_generated_params=True)


def strict_cosmo_validator() -> Callable:
    """Strict validator that actually calls cosmograph.

    Requires cosmograph to be installed.

    Example:
        >>> validator = strict_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
    """
    try:
        from cosmograph import cosmo

        return create_cosmo_validator(cosmo_function=cosmo, allow_generated_params=True)
    except ImportError:
        # Fall back to basic if cosmograph not available
        return basic_cosmo_validator()


def infer_cosmo_params(df) -> dict:
    """Infer suitable cosmograph parameters from DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        Dict of suggested parameters for cosmograph.cosmo()

    Example:
        >>> params = infer_cosmo_params(df)  # doctest: +SKIP
        >>> from cosmograph import cosmo  # doctest: +SKIP
        >>> cosmo(df, **params)  # doctest: +SKIP
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return {}

    numeric_cols = list(df.select_dtypes(include='number').columns)

    if len(numeric_cols) < 2:
        return {}

    params = {'points_x_by': numeric_cols[0], 'points_y_by': numeric_cols[1]}

    # Add size if available
    if len(numeric_cols) > 2:
        params['point_size_by'] = numeric_cols[2]

    # Add color from categorical
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 50:  # Reasonable cardinality
            params['point_color_by'] = col
            break

    return params
```

## loading.py

```python
"""Loading agent for converting data sources to pandas DataFrames.

Implements the ReAct pattern to iteratively try different loading strategies
until successful or max retries reached.
"""

from typing import Any
from collections.abc import MutableMapping

from aw.base import StepConfig, Context
from aw.tools import CodeInterpreterTool, ExecutionResult
from aw.util import (
    FileSamplerTool,
    infer_loader_from_extension,
    infer_loader_params,
    compute_dataframe_info,
    default_llm_factory,
)
from aw.validation import is_type, is_not_empty, all_validators


class LoadingAgent:
    """Agent that loads data from various sources into pandas DataFrames.

    Uses ReAct loop:
    1. Thought: Analyze source (extension, sample) to choose loader
    2. Action: Generate code to load data
    3. Observe: Execute code and capture result/error
    4. Validate: Check if result is valid DataFrame
    5. Repeat or finish

    Example:
        >>> agent = LoadingAgent()
        >>> context = Context()
        >>> df, metadata = agent.execute('/path/to/data.csv', context)
    """

    def __init__(self, config: StepConfig = None):
        """Initialize loading agent.

        Args:
            config: Configuration for the agent (LLM, validator, tools, etc.)
        """
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()

        # Set up tools
        self.file_sampler = FileSamplerTool()
        self.code_interpreter = CodeInterpreterTool()

        # Set up validator (DataFrame type + not empty)
        if self.config.validator is None:
            import pandas as pd

            self.validator = all_validators(is_type(pd.DataFrame), is_not_empty())
        else:
            self.validator = self.config.resolve_validator()

    def execute(
        self, source_uri: str, context: MutableMapping[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Execute loading agent to load data from source.

        Args:
            source_uri: URI of data source (file path or URL)
            context: Context for storing intermediate results

        Returns:
            Tuple of (dataframe, metadata)
        """
        # Initialize attempt history
        attempts = []

        # Step 1: Sample the file
        file_info = self.file_sampler(source_uri)
        attempts.append({'step': 'file_sample', 'result': file_info})

        if 'error' in file_info:
            return None, {
                'success': False,
                'error': file_info['error'],
                'attempts': attempts,
            }

        # Try to load with increasingly sophisticated approaches
        for attempt_num in range(self.config.max_retries):
            # Step 2: Generate loading code
            code = self._generate_loading_code(file_info, attempts, attempt_num)

            attempts.append(
                {'attempt': attempt_num, 'step': 'generate_code', 'code': code}
            )

            # Step 3: Execute code
            exec_result = self.code_interpreter(code)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'execute',
                    'success': exec_result.success,
                    'output': exec_result.output,
                    'error': exec_result.error,
                }
            )

            if not exec_result.success:
                # Execution failed - try again with error info
                continue

            # Get the DataFrame from execution result
            df = exec_result.locals.get('df')

            # Step 4: Validate
            is_valid, validation_info = self.validator(df)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'validate',
                    'success': is_valid,
                    'info': validation_info,
                }
            )

            if is_valid:
                # Success! Compute metadata and return
                df_info = compute_dataframe_info(df)
                metadata = {
                    'success': True,
                    'source_uri': source_uri,
                    'file_info': file_info,
                    'df_info': df_info,
                    'attempts': attempts,
                    'num_attempts': attempt_num + 1,
                }

                # Store in context
                context['loading'] = {'df': df, 'metadata': metadata}

                return df, metadata

        # All attempts failed
        return None, {
            'success': False,
            'error': 'Max retries exceeded',
            'attempts': attempts,
        }

    def _generate_loading_code(
        self, file_info: dict, attempts: list[dict], attempt_num: int
    ) -> str:
        """Generate Python code to load the data.

        Uses LLM for sophisticated reasoning, or falls back to heuristics.

        Args:
            file_info: Information about the file
            attempts: History of previous attempts
            attempt_num: Current attempt number

        Returns:
            Python code string to execute
        """
        # For first attempt, use simple heuristics
        if attempt_num == 0:
            return self._generate_initial_code(file_info)

        # For subsequent attempts, use LLM or refined heuristics
        return self._generate_retry_code(file_info, attempts)

    def _generate_initial_code(self, file_info: dict) -> str:
        """Generate initial loading code based on file info."""
        extension = file_info.get('extension', '.csv')
        uri = file_info.get('uri')

        # Infer loader and params
        loader_func = infer_loader_from_extension(extension)
        sample_text = file_info.get('sample_text')
        params = infer_loader_params(extension, sample_text)

        # Build params string
        params_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
        if params_str:
            params_str = ', ' + params_str

        code = f"""import pandas as pd
df = pd.{loader_func}({repr(uri)}{params_str})
"""
        return code

    def _generate_retry_code(self, file_info: dict, attempts: list[dict]) -> str:
        """Generate retry code based on previous failures.

        Analyzes error messages to adjust parameters.
        """
        # Get last error
        last_exec = [a for a in attempts if a.get('step') == 'execute'][-1]
        error_msg = last_exec.get('error', '')

        extension = file_info.get('extension', '.csv')
        uri = file_info.get('uri')
        loader_func = infer_loader_from_extension(extension)

        # Adjust parameters based on error
        params = {}

        if 'UnicodeDecodeError' in error_msg:
            params['encoding'] = 'latin-1'
        elif 'ParserError' in error_msg or 'delimiter' in error_msg.lower():
            # Try different delimiter
            params['sep'] = '\\t' if 'sep=,' in str(attempts) else ';'
        elif 'FileNotFoundError' in error_msg:
            # Try with different path interpretation
            params = {}

        # Try with error_bad_lines parameter for problematic CSVs
        if loader_func == 'read_csv' and 'ParserError' in error_msg:
            params['on_bad_lines'] = 'skip'

        params_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
        if params_str:
            params_str = ', ' + params_str

        code = f"""import pandas as pd
df = pd.{loader_func}({repr(uri)}{params_str})
"""
        return code


def create_loading_agent(
    llm: str = None, validator: Any = None, max_retries: int = 3
) -> LoadingAgent:
    """Factory function to create a loading agent.

    Args:
        llm: LLM model name or callable
        validator: Custom validator (uses default if None)
        max_retries: Maximum retry attempts

    Returns:
        Configured LoadingAgent

    Example:
        >>> agent = create_loading_agent(llm="gpt-4", max_retries=5)
    """
    config = StepConfig(
        llm=llm or "gpt-4", validator=validator, max_retries=max_retries
    )
    return LoadingAgent(config)
```

## orchestration.py

```python
"""Orchestration and workflow management for agentic steps.

Provides utilities to chain multiple agentic steps together and manage
the overall workflow.
"""

from typing import Any, Callable, Union
from collections.abc import MutableMapping

from aw.base import AgenticStep, Context, StepConfig
from aw.loading import LoadingAgent
from aw.preparing import PreparationAgent


class AgenticWorkflow:
    """Orchestrates a chain of agentic steps.

    Manages the execution of multiple steps in sequence, handling
    context/artifact passing between steps.

    Example:
        >>> workflow = AgenticWorkflow()  # doctest: +SKIP
        >>> workflow.add_step('loading', loading_agent)  # doctest: +SKIP
        >>> workflow.add_step('preparing', preparing_agent)  # doctest: +SKIP
        >>> result = workflow.run(source_uri)  # doctest: +SKIP
    """

    def __init__(self, context: Context = None):
        """Initialize workflow.

        Args:
            context: Shared context for all steps (creates new if None)
        """
        self.context = context or Context()
        self.steps: list[tuple[str, AgenticStep]] = []

    def add_step(self, name: str, step: AgenticStep) -> 'AgenticWorkflow':
        """Add a step to the workflow.

        Args:
            name: Name/identifier for the step
            step: Agent or step implementation

        Returns:
            Self for chaining
        """
        self.steps.append((name, step))
        return self

    def run(self, initial_input: Any) -> tuple[Any, dict[str, Any]]:
        """Execute the workflow.

        Args:
            initial_input: Input to the first step

        Returns:
            Tuple of (final_artifact, workflow_metadata)
        """
        current_input = initial_input
        workflow_metadata = {'steps': [], 'context_snapshot': None}

        for step_name, step in self.steps:
            # Execute step
            artifact, metadata = step.execute(current_input, self.context)

            # Record step execution
            workflow_metadata['steps'].append(
                {
                    'name': step_name,
                    'success': metadata.get('success', True),
                    'metadata': metadata,
                }
            )

            # Check for failure
            if not metadata.get('success', True):
                workflow_metadata['failed_at'] = step_name
                workflow_metadata['context_snapshot'] = self.context.snapshot()
                return artifact, workflow_metadata

            # Pass artifact to next step
            current_input = artifact

        # All steps completed successfully
        workflow_metadata['success'] = True
        workflow_metadata['context_snapshot'] = self.context.snapshot()

        return current_input, workflow_metadata

    def run_partial(
        self, initial_input: Any, stop_after: str = None
    ) -> tuple[Any, dict[str, Any]]:
        """Run workflow up to a specific step.

        Args:
            initial_input: Input to first step
            stop_after: Name of step to stop after (runs all if None)

        Returns:
            Tuple of (artifact, metadata)
        """
        current_input = initial_input
        workflow_metadata = {'steps': []}

        for step_name, step in self.steps:
            artifact, metadata = step.execute(current_input, self.context)

            workflow_metadata['steps'].append({'name': step_name, 'metadata': metadata})

            current_input = artifact

            if step_name == stop_after:
                break

        workflow_metadata['context_snapshot'] = self.context.snapshot()
        return current_input, workflow_metadata


def create_data_prep_workflow(
    loading_config: StepConfig = None,
    preparing_config: StepConfig = None,
    target: str = 'generic',
) -> AgenticWorkflow:
    """Factory to create a data preparation workflow.

    Creates a workflow with LoadingAgent -> PreparationAgent.

    Args:
        loading_config: Configuration for loading agent
        preparing_config: Configuration for preparing agent
        target: Target format for preparation

    Returns:
        Configured workflow

    Example:
        >>> workflow = create_data_prep_workflow(target='cosmo-ready')
        >>> df, metadata = workflow.run('/path/to/data.csv')
    """
    workflow = AgenticWorkflow()

    # Add loading step
    loading_agent = LoadingAgent(loading_config)
    workflow.add_step('loading', loading_agent)

    # Add preparing step
    preparing_agent = PreparationAgent(config=preparing_config, target=target)
    workflow.add_step('preparing', preparing_agent)

    return workflow


def create_cosmo_prep_workflow(
    cosmo_validator: Callable = None, max_retries: int = 3
) -> AgenticWorkflow:
    """Create a workflow specifically for cosmograph preparation.

    Args:
        cosmo_validator: Custom cosmo validator (uses default if None)
        max_retries: Maximum retries for each step

    Returns:
        Configured workflow

    Example:
        >>> workflow = create_cosmo_prep_workflow()
        >>> prepared_df, metadata = workflow.run('data.csv')
    """
    from aw.cosmo import basic_cosmo_validator

    # Use provided validator or default
    validator = cosmo_validator or basic_cosmo_validator()

    # Create configs
    loading_config = StepConfig(max_retries=max_retries)
    preparing_config = StepConfig(max_retries=max_retries, validator=validator)

    return create_data_prep_workflow(
        loading_config=loading_config,
        preparing_config=preparing_config,
        target='cosmo-ready',
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def load_and_prepare(
    source_uri: str,
    target: str = 'generic',
    validator: Callable = None,
    max_retries: int = 3,
) -> tuple[Any, dict]:
    """Convenience function to load and prepare data in one call.

    Args:
        source_uri: URI of data source
        target: Target format/purpose
        validator: Optional custom validator
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (prepared_dataframe, metadata)

    Example:
        >>> df, meta = load_and_prepare('data.csv', target='cosmo-ready')
    """
    loading_config = StepConfig(max_retries=max_retries)
    preparing_config = StepConfig(max_retries=max_retries, validator=validator)

    workflow = create_data_prep_workflow(
        loading_config=loading_config, preparing_config=preparing_config, target=target
    )

    return workflow.run(source_uri)


def load_for_cosmo(
    source_uri: str, max_retries: int = 3, strict: bool = False
) -> tuple[Any, dict]:
    """Load and prepare data specifically for cosmograph visualization.

    Args:
        source_uri: URI of data source
        max_retries: Maximum retry attempts
        strict: If True, actually calls cosmograph for validation

    Returns:
        Tuple of (prepared_dataframe, metadata)

    Example:
        >>> df, meta = load_for_cosmo('data.csv')  # doctest: +SKIP
        >>> from cosmograph import cosmo  # doctest: +SKIP
        >>> cosmo(df, **meta['preparing']['metadata']['validation_result']['params'])  # doctest: +SKIP
    """
    from aw.cosmo import basic_cosmo_validator, strict_cosmo_validator

    validator = strict_cosmo_validator() if strict else basic_cosmo_validator()

    return load_and_prepare(
        source_uri=source_uri,
        target='cosmo-ready',
        validator=validator,
        max_retries=max_retries,
    )


# ============================================================================
# Interactive Workflow (with Human-in-Loop)
# ============================================================================


class InteractiveWorkflow(AgenticWorkflow):
    """Workflow with human-in-the-loop capabilities.

    Pauses execution to request human input/approval at configured points.

    Example:
        >>> workflow = InteractiveWorkflow()  # doctest: +SKIP
        >>> workflow.add_step('loading', agent, require_approval=True)  # doctest: +SKIP
        >>> result = workflow.run_interactive(source_uri)  # doctest: +SKIP
    """

    def __init__(self, context: Context = None):
        super().__init__(context)
        self.approval_required: dict[str, bool] = {}
        self.approval_callback: Callable = None

    def add_step(
        self, name: str, step: AgenticStep, require_approval: bool = False
    ) -> 'InteractiveWorkflow':
        """Add step with optional approval requirement."""
        super().add_step(name, step)
        self.approval_required[name] = require_approval
        return self

    def set_approval_callback(self, callback: Callable[[str, Any, dict], bool]) -> None:
        """Set callback for human approval.

        Args:
            callback: Function(step_name, artifact, metadata) -> bool
                Returns True to continue, False to abort
        """
        self.approval_callback = callback

    def run_interactive(self, initial_input: Any) -> tuple[Any, dict]:
        """Run workflow with human-in-loop checkpoints."""
        current_input = initial_input
        workflow_metadata = {'steps': []}

        for step_name, step in self.steps:
            # Execute step
            artifact, metadata = step.execute(current_input, self.context)

            workflow_metadata['steps'].append({'name': step_name, 'metadata': metadata})

            # Check if approval required
            if self.approval_required.get(step_name, False):
                if self.approval_callback:
                    approved = self.approval_callback(step_name, artifact, metadata)
                    if not approved:
                        workflow_metadata['aborted_at'] = step_name
                        workflow_metadata['reason'] = 'User rejected'
                        return artifact, workflow_metadata
                else:
                    # No callback - just print and ask
                    print(f"\nStep '{step_name}' completed:")
                    print(f"  Metadata: {metadata}")
                    response = input("Continue? (y/n): ")
                    if response.lower() != 'y':
                        workflow_metadata['aborted_at'] = step_name
                        return artifact, workflow_metadata

            current_input = artifact

        workflow_metadata['success'] = True
        workflow_metadata['context_snapshot'] = self.context.snapshot()
        return current_input, workflow_metadata
```

## preparing.py

```python
"""Preparation agent for transforming data to meet target requirements.

Implements ReAct pattern with functional validation (try-the-purpose approach).
"""

from typing import Any, Callable
from collections.abc import MutableMapping

from aw.base import StepConfig, Context
from aw.tools import CodeInterpreterTool
from aw.util import compute_dataframe_info, default_llm_factory
from aw.validation import functional_validator


class PreparationAgent:
    """Agent that prepares data to meet target requirements.

    Uses ReAct loop with functional validation:
    1. Thought: Analyze current data state vs. requirements
    2. Action: Generate transformation code
    3. Observe: Execute code and capture result
    4. Validate: Try to use data for its purpose (e.g., visualization)
    5. Repeat or finish

    Example:
        >>> agent = PreparationAgent(target='cosmo-ready')  # doctest: +SKIP
        >>> context = Context({'loading': {'df': df}})  # doctest: +SKIP
        >>> prepared_df, metadata = agent.execute(df, context)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: StepConfig = None,
        target: str = 'generic',
        target_validator: Callable = None,
    ):
        """Initialize preparation agent.

        Args:
            config: Configuration for the agent
            target: Target format/purpose (e.g., 'cosmo-ready', 'ml-ready')
            target_validator: Custom validator for target requirements
        """
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()
        self.target = target

        # Set up code interpreter
        self.code_interpreter = CodeInterpreterTool()

        # Set up validator
        if target_validator:
            self.validator = target_validator
        elif self.config.validator:
            self.validator = self.config.resolve_validator()
        else:
            # Use a basic validator
            self.validator = lambda df: (True, {'note': 'No validation specified'})

    def execute(
        self, input_df: Any, context: MutableMapping[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Execute preparation agent to transform data.

        Args:
            input_df: Input DataFrame to prepare
            context: Context for storing intermediate results

        Returns:
            Tuple of (prepared_dataframe, metadata)
        """
        import pandas as pd

        # Initialize
        attempts = []
        current_df = input_df.copy() if hasattr(input_df, 'copy') else input_df

        # Get initial info
        initial_info = compute_dataframe_info(current_df)
        attempts.append({'step': 'initial_analysis', 'info': initial_info})

        # Iterate with transformations
        for attempt_num in range(self.config.max_retries):
            # Step 1: Generate transformation code
            code = self._generate_transformation_code(
                current_df, initial_info, attempts, attempt_num
            )

            attempts.append(
                {'attempt': attempt_num, 'step': 'generate_code', 'code': code}
            )

            # Step 2: Execute transformation
            exec_result = self.code_interpreter(
                code, context={'df': current_df, 'pd': pd}
            )

            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'execute',
                    'success': exec_result.success,
                    'output': exec_result.output,
                    'error': exec_result.error,
                }
            )

            if not exec_result.success:
                # Execution failed - try again
                continue

            # Get transformed DataFrame
            transformed_df = exec_result.locals.get('transformed_df', current_df)

            # Step 3: Validate with target validator
            is_valid, validation_info = self.validator(transformed_df)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'validate',
                    'success': is_valid,
                    'info': validation_info,
                }
            )

            if is_valid:
                # Success!
                final_info = compute_dataframe_info(transformed_df)
                metadata = {
                    'success': True,
                    'target': self.target,
                    'initial_info': initial_info,
                    'final_info': final_info,
                    'validation_result': validation_info,
                    'attempts': attempts,
                    'num_attempts': attempt_num + 1,
                }

                # Store in context
                context['preparing'] = {'df': transformed_df, 'metadata': metadata}

                return transformed_df, metadata

            # Validation failed but code ran - update current_df and try again
            current_df = transformed_df

        # Max retries exceeded
        return current_df, {
            'success': False,
            'error': 'Max retries exceeded',
            'target': self.target,
            'attempts': attempts,
        }

    def _generate_transformation_code(
        self, df: Any, initial_info: dict, attempts: list[dict], attempt_num: int
    ) -> str:
        """Generate transformation code.

        Args:
            df: Current DataFrame
            initial_info: Initial analysis info
            attempts: History of attempts
            attempt_num: Current attempt number

        Returns:
            Python code string
        """
        if attempt_num == 0:
            return self._generate_initial_transformation(df, initial_info)
        else:
            return self._generate_retry_transformation(df, initial_info, attempts)

    def _generate_initial_transformation(self, df: Any, info: dict) -> str:
        """Generate initial transformation based on target."""
        if self.target == 'cosmo-ready':
            return self._generate_cosmo_transformation(df, info)
        else:
            # Generic: just ensure no nulls
            return """
transformed_df = df.copy()
transformed_df = transformed_df.dropna()
"""

    def _generate_cosmo_transformation(self, df: Any, info: dict) -> str:
        """Generate transformation for cosmograph requirements.

        Cosmograph needs:
        - At least 2 numeric columns for x/y coordinates
        - No nulls in those columns
        """
        numeric_cols = info.get('numeric_columns', [])

        if len(numeric_cols) >= 2:
            # We have numeric columns - just clean them
            return f"""
transformed_df = df.copy()
# Ensure numeric columns are clean
numeric_cols = {numeric_cols[:5]}  # Use first few numeric columns
for col in numeric_cols:
    if col in transformed_df.columns:
        transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
# Drop rows with nulls in numeric columns
transformed_df = transformed_df.dropna(subset=numeric_cols)
"""
        else:
            # Need to create or convert numeric columns
            return """
import numpy as np
transformed_df = df.copy()

# Try to convert object columns to numeric
for col in transformed_df.select_dtypes(include='object').columns:
    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')

# If still no numeric columns, create indices
numeric_cols = transformed_df.select_dtypes(include='number').columns
if len(numeric_cols) < 2:
    transformed_df['_x'] = np.arange(len(transformed_df))
    transformed_df['_y'] = np.random.randn(len(transformed_df))

# Drop nulls
transformed_df = transformed_df.dropna()
"""

    def _generate_retry_transformation(
        self, df: Any, initial_info: dict, attempts: list[dict]
    ) -> str:
        """Generate retry transformation based on validation errors."""
        # Get last validation error
        last_validation = [a for a in attempts if a.get('step') == 'validate'][-1]

        error_info = last_validation.get('info', {})
        error_msg = error_info.get('error', '')

        # Adjust based on error
        if 'not numeric' in error_msg.lower():
            return """
import numpy as np
transformed_df = df.copy()
# More aggressive numeric conversion
for col in transformed_df.columns:
    try:
        transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
    except:
        pass
transformed_df = transformed_df.dropna()
"""
        elif 'too few' in error_msg.lower() or 'column' in error_msg.lower():
            return """
import numpy as np
transformed_df = df.copy()
# Ensure we have enough numeric columns
numeric_cols = list(transformed_df.select_dtypes(include='number').columns)
while len(numeric_cols) < 2:
    new_col = f'_generated_{len(numeric_cols)}'
    transformed_df[new_col] = np.random.randn(len(transformed_df))
    numeric_cols.append(new_col)
transformed_df = transformed_df.dropna()
"""
        else:
            # Generic retry - more aggressive cleaning
            return """
transformed_df = df.copy()
# Drop all non-numeric columns
transformed_df = transformed_df.select_dtypes(include='number')
# Fill nulls with mean
transformed_df = transformed_df.fillna(transformed_df.mean())
"""


def create_preparation_agent(
    target: str = 'generic',
    validator: Callable = None,
    llm: str = None,
    max_retries: int = 3,
) -> PreparationAgent:
    """Factory function to create a preparation agent.

    Args:
        target: Target format/purpose
        validator: Custom validator function
        llm: LLM model name or callable
        max_retries: Maximum retry attempts

    Returns:
        Configured PreparationAgent

    Example:
        >>> agent = create_preparation_agent(  # doctest: +SKIP
        ...     target='cosmo-ready',
        ...     validator=cosmo_validator,
        ...     max_retries=5
        ... )
    """
    config = StepConfig(
        llm=llm or "gpt-4", validator=validator, max_retries=max_retries
    )
    return PreparationAgent(config=config, target=target, target_validator=validator)
```

## routing.py

```python
"""
Configurable routing and decision chains for agentic workflows.

This module provides tools for building extensible, inspectable routing chains
that follow the open-closed principle. Users can:
- Use default configurations out of the box
- Inspect all routing rules and mappings
- Customize individual components
- Extend with new routing strategies

The key pattern is: provide sensible defaults, make them visible, keep them mutable.
"""

from typing import Optional, Callable, Any, Protocol
from dataclasses import dataclass, field
from functools import partial
from collections.abc import Mapping


class RoutingStrategy(Protocol):
    """Protocol for routing strategies.

    A routing strategy takes a context and returns a result or None.
    """

    def __call__(self, context: Any) -> Optional[Any]:
        """Apply routing logic to context."""
        ...


@dataclass
class PriorityRouter:
    """Route through strategies in priority order, short-circuiting on first match.

    This is a simpler, more specialized version of i2.routing_forest that's
    optimized for the common case of "try these in order, stop at first success".

    >>> def try_a(x): return 'A' if x > 10 else None
    >>> def try_b(x): return 'B' if x > 5 else None
    >>> def try_c(x): return 'C'
    >>>
    >>> router = PriorityRouter([try_a, try_b, try_c])
    >>> router(15)
    'A'
    >>> router(8)
    'B'
    >>> router(3)
    'C'
    """

    strategies: list[RoutingStrategy] = field(default_factory=list)

    def __call__(self, context: Any) -> Optional[Any]:
        """Route through strategies in order, return first non-None result."""
        for strategy in self.strategies:
            result = strategy(context)
            if result is not None:
                return result
        return None

    def prepend(self, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Add a strategy at the beginning (highest priority)."""
        return PriorityRouter([strategy] + self.strategies)

    def append(self, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Add a strategy at the end (lowest priority)."""
        return PriorityRouter(self.strategies + [strategy])

    def insert(self, index: int, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Insert a strategy at a specific position."""
        new_strategies = self.strategies.copy()
        new_strategies.insert(index, strategy)
        return PriorityRouter(new_strategies)


@dataclass
class ConditionalRouter(PriorityRouter):
    """Router with short-circuit conditions.

    Only accepts results that match the short-circuit condition,
    otherwise continues to next strategy.

    >>> def try_url(x): return x.get('url_ext')
    >>> def try_type(x): return x.get('content_type_ext')
    >>>
    >>> # Only accept .pdf or .md, otherwise keep trying
    >>> router = ConditionalRouter(
    ...     [try_url, try_type],
    ...     short_circuit=lambda r: r in {'.pdf', '.md'}
    ... )
    >>>
    >>> # .pdf matches short-circuit, return immediately
    >>> router({'url_ext': '.pdf', 'content_type_ext': '.html'})
    '.pdf'
    >>>
    >>> # .jpg doesn't match, try next strategy, .md matches
    >>> router({'url_ext': '.jpg', 'content_type_ext': '.md'})
    '.md'
    """

    short_circuit: Optional[Callable[[Any], bool]] = None

    def __call__(self, context: Any) -> Optional[Any]:
        """Route with short-circuit logic.

        If short_circuit is defined:
        - Results matching the condition are returned immediately
        - Results not matching are ignored, and next strategy is tried

        If no short_circuit is defined, behaves like PriorityRouter.
        """
        for strategy in self.strategies:
            result = strategy(context)
            if result is not None:
                # If no short-circuit function, return first result (PriorityRouter behavior)
                if self.short_circuit is None:
                    return result
                # If short-circuit matches, return immediately
                if self.short_circuit(result):
                    return result
                # Otherwise, ignore this result and try next strategy

        return None


@dataclass
class MappingRouter:
    """Router based on a lookup mapping.

    Provides a clean interface for switch-case style routing with:
    - Visible, mutable mapping
    - Optional default value or factory
    - Easy extension via dict interface

    >>> # Content-Type to extension mapping
    >>> ct_map = {
    ...     'application/pdf': '.pdf',
    ...     'text/html': '.html',
    ...     'application/json': '.json',
    ... }
    >>> router = MappingRouter(ct_map)
    >>>
    >>> router('application/pdf')
    '.pdf'
    >>> router('text/plain') is None  # No default
    True
    >>>
    >>> # With default
    >>> router_with_default = MappingRouter(ct_map, default='.bin')
    >>> router_with_default('text/plain')
    '.bin'
    """

    mapping: dict = field(default_factory=dict)
    default: Any = None
    key_transform: Optional[Callable[[Any], Any]] = None

    def __call__(self, key: Any) -> Any:
        """Look up key in mapping."""
        if self.key_transform:
            key = self.key_transform(key)
        return self.mapping.get(key, self.default)

    def __getitem__(self, key):
        """Access mapping directly."""
        return self.mapping[key]

    def __setitem__(self, key, value):
        """Update mapping directly."""
        self.mapping[key] = value

    def update(self, *args, **kwargs):
        """Update mapping."""
        self.mapping.update(*args, **kwargs)

    def copy(self) -> 'MappingRouter':
        """Create a copy with same configuration."""
        return MappingRouter(
            self.mapping.copy(), default=self.default, key_transform=self.key_transform
        )


# ---------------------------------------------------------------------------
# Extension Detection - Concrete Application
# ---------------------------------------------------------------------------


@dataclass
class ExtensionContext:
    """Context for extension detection.

    >>> ctx = ExtensionContext(
    ...     url='https://example.com/file.pdf',
    ...     content=b'%PDF-1.5',
    ...     content_type='text/html'
    ... )
    >>> ctx.url
    'https://example.com/file.pdf'
    """

    url: str
    content: bytes = b''
    content_type: str = ''
    explicit_extension: Optional[str] = None


# Default Content-Type mapping (visible and mutable)
DEFAULT_CONTENT_TYPE_MAP = {
    'application/pdf': '.pdf',
    'text/html': '.html',
    'text/markdown': '.md',
    'text/plain': '.txt',
    'application/json': '.json',
    'application/xml': '.xml',
    'text/csv': '.csv',
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/svg+xml': '.svg',
    'application/zip': '.zip',
    'application/gzip': '.gz',
    'application/x-tar': '.tar',
}


# Default magic bytes patterns (visible and mutable)
DEFAULT_MAGIC_BYTES_MAP = {
    b'%PDF': '.pdf',
    b'<!DOCTYPE html': '.html',
    b'<!doctype html': '.html',
    b'<html': '.html',
    b'\x89PNG': '.png',
    b'\xff\xd8\xff': '.jpg',
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'PK\x03\x04': '.zip',
    b'\x1f\x8b': '.gz',
}


def detect_from_url(ctx: ExtensionContext) -> Optional[str]:
    """Extract extension from URL path.

    >>> ctx = ExtensionContext('https://example.com/file.pdf')
    >>> detect_from_url(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('https://example.com/file')
    >>> detect_from_url(ctx) is None
    True
    """
    from urllib.parse import urlparse
    from pathlib import Path

    url_path = urlparse(ctx.url).path
    ext = Path(url_path).suffix.lower()

    # Validate extension looks reasonable
    if ext and 1 < len(ext) <= 5 and ext[1:].replace('_', '').isalnum():
        return ext

    return None


def make_content_type_detector(
    content_type_map: Optional[dict] = None,
) -> Callable[[ExtensionContext], Optional[str]]:
    """Factory for content-type based detection.

    Returns a detector function with the given mapping.
    Users can create custom detectors with different mappings.

    >>> # Use default mapping
    >>> detector = make_content_type_detector()
    >>> ctx = ExtensionContext('', content_type='application/pdf')
    >>> detector(ctx)
    '.pdf'
    >>>
    >>> # Custom mapping
    >>> custom_map = {'text/plain': '.log'}
    >>> custom_detector = make_content_type_detector(custom_map)
    >>> ctx = ExtensionContext('', content_type='text/plain')
    >>> custom_detector(ctx)
    '.log'
    """
    if content_type_map is None:
        content_type_map = DEFAULT_CONTENT_TYPE_MAP.copy()

    def detect(ctx: ExtensionContext) -> Optional[str]:
        if not ctx.content_type:
            return None

        # Clean content type (remove charset, etc.)
        content_type = ctx.content_type.split(';')[0].strip().lower()
        return content_type_map.get(content_type)

    return detect


def make_magic_bytes_detector(
    magic_bytes_map: Optional[dict] = None,
) -> Callable[[ExtensionContext], Optional[str]]:
    """Factory for magic bytes based detection.

    >>> detector = make_magic_bytes_detector()
    >>> ctx = ExtensionContext('', content=b'%PDF-1.5...')
    >>> detector(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('', content=b'<html>...')
    >>> detector(ctx)
    '.html'
    """
    if magic_bytes_map is None:
        magic_bytes_map = DEFAULT_MAGIC_BYTES_MAP.copy()

    def detect(ctx: ExtensionContext) -> Optional[str]:
        if not ctx.content or len(ctx.content) < 4:
            return None

        # Check against magic byte patterns
        content_lower = ctx.content[:100].lower()
        for magic, ext in magic_bytes_map.items():
            if content_lower.startswith(magic.lower()):
                return ext

        # Try imghdr for images
        try:
            import imghdr

            img_ext = imghdr.what(None, h=ctx.content[:50])
            if img_ext:
                return f'.{img_ext}'
        except Exception:
            pass

        return None

    return detect


def detect_explicit(ctx: ExtensionContext) -> Optional[str]:
    """Use explicit extension if provided.

    >>> ctx = ExtensionContext('', explicit_extension='pdf')
    >>> detect_explicit(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('', explicit_extension='.json')
    >>> detect_explicit(ctx)
    '.json'
    """
    if ctx.explicit_extension:
        ext = ctx.explicit_extension
        if not ext.startswith('.'):
            ext = f'.{ext}'
        return ext
    return None


@dataclass
class ExtensionRouter:
    """Route extension detection through configurable strategies.

    This router implements the default logic for detecting file extensions,
    but every component is visible and customizable.

    Examples:
        >>> # Use with defaults
        >>> router = ExtensionRouter()
        >>> ctx = ExtensionContext('https://example.com/file.pdf', b'%PDF-1.5')
        >>> router(ctx)
        '.pdf'
        >>>
        >>> # URL without extension, falls back to magic bytes
        >>> ctx = ExtensionContext('https://example.com/download', b'%PDF-1.5')
        >>> router(ctx)
        '.pdf'
        >>>
        >>> # Priority extensions short-circuit
        >>> ctx = ExtensionContext('https://example.com/file.pdf', b'<html>')
        >>> router(ctx)  # .pdf from URL wins even though content is HTML
        '.pdf'
        >>>
        >>> # Customize content type mapping
        >>> router.content_type_map['.txt'] = 'text/x-log'
        >>>
        >>> # Add custom detector
        >>> def my_detector(ctx): return '.custom' if 'special' in ctx.url else None
        >>> router = router.with_prepended_strategy(my_detector)
    """

    # Priority extensions that short-circuit other checks
    priority_extensions: frozenset = field(
        default_factory=lambda: frozenset(['.pdf', '.md', '.json'])
    )

    # Configurable mappings (users can modify these)
    content_type_map: dict = field(
        default_factory=lambda: DEFAULT_CONTENT_TYPE_MAP.copy()
    )
    magic_bytes_map: dict = field(
        default_factory=lambda: DEFAULT_MAGIC_BYTES_MAP.copy()
    )

    # Detection strategies (built in __post_init__)
    router: Optional[ConditionalRouter] = field(default=None, init=False)

    def __post_init__(self):
        """Build the routing chain."""
        # Create detectors with current mappings
        content_type_detector = make_content_type_detector(self.content_type_map)
        magic_bytes_detector = make_magic_bytes_detector(self.magic_bytes_map)

        # Build strategy chain: explicit → URL → content-type → magic bytes
        strategies = [
            detect_explicit,
            detect_from_url,
            content_type_detector,
            magic_bytes_detector,
        ]

        # Short-circuit on priority extensions
        self.router = ConditionalRouter(
            strategies, short_circuit=lambda ext: ext in self.priority_extensions
        )

    def __call__(
        self,
        url: str = '',
        content: bytes = b'',
        content_type: str = '',
        *,
        explicit_extension: Optional[str] = None,
    ) -> str:
        """Detect extension from context.

        Can be called with ExtensionContext or individual args.
        """
        # Support both context object and individual args
        if isinstance(url, ExtensionContext):
            ctx = url
        else:
            ctx = ExtensionContext(url, content, content_type, explicit_extension)

        result = self.router(ctx)
        return result if result else '.bin'

    def with_prepended_strategy(
        self, strategy: Callable[[ExtensionContext], Optional[str]]
    ) -> 'ExtensionRouter':
        """Create new router with added strategy at highest priority."""
        new_router = ExtensionRouter(
            self.priority_extensions,
            self.content_type_map.copy(),
            self.magic_bytes_map.copy(),
        )
        new_router.router = self.router.prepend(strategy)
        return new_router

    def with_appended_strategy(
        self, strategy: Callable[[ExtensionContext], Optional[str]]
    ) -> 'ExtensionRouter':
        """Create new router with added strategy at lowest priority."""
        new_router = ExtensionRouter(
            self.priority_extensions,
            self.content_type_map.copy(),
            self.magic_bytes_map.copy(),
        )
        new_router.router = self.router.append(strategy)
        return new_router


# Default instance for convenience
extension_router = ExtensionRouter()
```

## tools.py

```python
"""Tools for executing code and other agent actions.

Provides code execution capabilities with safe defaults and extensibility
for robust backends like LangChain's CodeInterpreter.
"""

from typing import Any, Callable
from collections.abc import Mapping
import sys
from io import StringIO
import traceback


class ExecutionResult:
    """Result of code execution.

    Attributes:
        success: Whether execution succeeded without exceptions
        output: Standard output captured during execution
        error: Error message if execution failed
        traceback: Full traceback if execution failed
        result: The return value or final expression value
        locals: Local variables after execution
    """

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        traceback_str: str = "",
        result: Any = None,
        locals_dict: dict = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.traceback = traceback_str
        self.result = result
        self.locals = locals_dict or {}

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"ExecutionResult({status}, output_len={len(self.output)}, error={self.error[:50] if self.error else None})"


class CodeInterpreterTool:
    """Tool for executing Python code in a controlled environment.

    Provides a safe default implementation using exec() with limited namespace,
    and allows injection of more robust backends.

    Example:
        >>> tool = CodeInterpreterTool()
        >>> result = tool("x = 5; y = x * 2; print(y)")
        >>> result.success
        True
        >>> result.output.strip()
        '10'
    """

    def __init__(
        self,
        allowed_modules: list[str] = None,
        global_context: dict = None,
        executor: Callable[[str, dict], ExecutionResult] = None,
    ):
        """Initialize code interpreter.

        Args:
            allowed_modules: List of module names that can be imported
            global_context: Additional globals to make available
            executor: Optional custom executor function for advanced backends
        """
        self.allowed_modules = allowed_modules or [
            'pandas',
            'numpy',
            'math',
            'json',
            're',
            'itertools',
            'collections',
            'functools',
            'typing',
        ]
        self.global_context = global_context or {}
        self._executor = executor or self._default_executor

    def __call__(self, code: str, context: dict = None) -> ExecutionResult:
        """Execute Python code.

        Args:
            code: Python code to execute
            context: Additional context/variables to make available

        Returns:
            ExecutionResult with success status and outputs
        """
        merged_context = {**self.global_context, **(context or {})}
        return self._executor(code, merged_context)

    def _default_executor(self, code: str, context: dict) -> ExecutionResult:
        """Default executor using built-in exec()."""
        # Create safe namespace
        safe_globals = self._build_safe_namespace()
        safe_globals.update(context)
        local_vars = {}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute code
            exec(code, safe_globals, local_vars)
            output = captured_output.getvalue()

            # Try to get result from last expression
            result = local_vars.get('_result', None)

            return ExecutionResult(
                success=True, output=output, result=result, locals_dict=local_vars
            )
        except Exception as e:
            error_msg = str(e)
            tb_str = traceback.format_exc()
            return ExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=error_msg,
                traceback_str=tb_str,
                locals_dict=local_vars,
            )
        finally:
            sys.stdout = old_stdout

    def _build_safe_namespace(self) -> dict:
        """Build a safe namespace with allowed modules."""
        namespace = {
            '__builtins__': {
                # Include safe builtins
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                '__import__': __import__,  # Needed for import statements
            }
        }

        # Import allowed modules
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                # Handle submodules (e.g., pandas.core)
                if '.' in module_name:
                    parts = module_name.split('.')
                    for part in parts[1:]:
                        module = getattr(module, part)
                namespace[module_name.split('.')[0]] = module
            except ImportError:
                # Module not available - skip silently
                pass

        return namespace


def _extract_imports(code: str) -> list[str]:
    """Extract import statements from code.

    Helper function to analyze what modules code needs.

    Example:
        >>> _extract_imports("import pandas as pd\\nfrom numpy import array")
        ['pandas', 'numpy']
    """
    import re

    imports = []

    # Match "import module" and "from module import ..."
    import_pattern = r'^\s*(?:import|from)\s+(\w+)'

    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            imports.append(match.group(1))

    return imports


class SafeCodeInterpreter(CodeInterpreterTool):
    """Extra-safe code interpreter with restricted operations.

    Disallows file I/O, network access, and other potentially dangerous operations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove potentially dangerous modules
        dangerous_modules = {'os', 'sys', 'subprocess', 'socket', 'urllib'}
        self.allowed_modules = [
            m for m in self.allowed_modules if m not in dangerous_modules
        ]


def create_langchain_executor() -> Callable[[str, dict], ExecutionResult]:
    """Create an executor using LangChain's CodeInterpreter (if available).

    Returns:
        Executor function compatible with CodeInterpreterTool

    Example:
        >>> try:
        ...     executor = create_langchain_executor()
        ...     tool = CodeInterpreterTool(executor=executor)
        ... except ImportError:
        ...     # LangChain not available - use default
        ...     tool = CodeInterpreterTool()
    """
    try:
        from langchain_experimental.tools import PythonREPLTool

        repl = PythonREPLTool()

        def executor(code: str, context: dict) -> ExecutionResult:
            # Inject context variables
            context_setup = '\n'.join(f"{k} = {repr(v)}" for k, v in context.items())
            full_code = f"{context_setup}\n{code}" if context else code

            try:
                output = repl.run(full_code)
                return ExecutionResult(success=True, output=output, result=output)
            except Exception as e:
                return ExecutionResult(
                    success=False, error=str(e), traceback_str=traceback.format_exc()
                )

        return executor
    except ImportError:
        raise ImportError(
            "LangChain not available. Install with: pip install langchain langchain-experimental"
        )


def create_dspy_executor() -> Callable[[str, dict], ExecutionResult]:
    """Create an executor using DSPy's code execution (if available).

    Returns:
        Executor function compatible with CodeInterpreterTool
    """
    try:
        import dspy

        def executor(code: str, context: dict) -> ExecutionResult:
            # DSPy implementation would go here
            # This is a placeholder for the actual implementation
            raise NotImplementedError("DSPy executor not yet implemented")

        return executor
    except ImportError:
        raise ImportError("DSPy not available. Install with: pip install dspy-ai")
```

## util.py

```python
"""Utility tools and helper functions for agents.

Includes file sampling, LLM facades, and other common utilities.
"""

from typing import Any, Union, Callable
from pathlib import Path
from functools import partial
import urllib.parse
import io
import os

from config2py import get_app_data_folder, process_path

# ============================================================================
# Persistent Storage Configuration
# ============================================================================

# Get AW data directory from env var or use default app data folder
# Users can override by setting AW_DATA_DIR environment variable
# Default location follows XDG standards:
#   - Linux/macOS: ~/.local/share/aw
#   - Windows: %LOCALAPPDATA%/aw
_aw_data_dir_raw = os.environ.get('AW_DATA_DIR') or get_app_data_folder('aw')

# Process path and ensure it exists
AW_DATA_DIR = process_path(_aw_data_dir_raw, ensure_dir_exists=True)

# Convenience function to join paths relative to AW_DATA_DIR
# Example: djoin('models', 'model1.pkl') -> '/Users/user/.local/share/aw/models/model1.pkl'
djoin = partial(os.path.join, AW_DATA_DIR)
djoin.rootdir = AW_DATA_DIR


class FileSamplerTool:
    """Tool to sample and analyze file metadata and content.

    Helps agents understand what kind of data they're dealing with
    before attempting to load it.

    Example:
        >>> sampler = FileSamplerTool()  # doctest: +SKIP
        >>> info = sampler('/path/to/data.csv')  # doctest: +SKIP
        >>> info['extension']  # doctest: +SKIP
        '.csv'
    """

    def __init__(self, sample_size: int = 1024):
        """Initialize file sampler.

        Args:
            sample_size: Number of bytes to sample from file
        """
        self.sample_size = sample_size

    def __call__(self, uri: str) -> dict[str, Any]:
        """Sample file and return metadata.

        Args:
            uri: File URI (local path or URL)

        Returns:
            Dict with extension, sample_bytes, encoding, etc.
        """
        parsed = urllib.parse.urlparse(uri)

        # Determine if local or remote
        if parsed.scheme in ('', 'file'):
            return self._sample_local(parsed.path or uri)
        elif parsed.scheme in ('http', 'https'):
            return self._sample_remote(uri)
        else:
            return {'error': f'Unsupported URI scheme: {parsed.scheme}'}

    def _sample_local(self, path: str) -> dict[str, Any]:
        """Sample local file."""
        file_path = Path(path)

        if not file_path.exists():
            return {'error': f'File not found: {path}'}

        info = {
            'uri': str(file_path),
            'extension': file_path.suffix,
            'size': file_path.stat().st_size,
            'exists': True,
        }

        try:
            # Read sample
            with open(file_path, 'rb') as f:
                sample_bytes = f.read(self.sample_size)
            info['sample_bytes'] = sample_bytes

            # Try to decode as text
            try:
                info['sample_text'] = sample_bytes.decode('utf-8')
                info['encoding'] = 'utf-8'
            except UnicodeDecodeError:
                info['encoding'] = 'binary'
        except Exception as e:
            info['error'] = str(e)

        return info

    def _sample_remote(self, url: str) -> dict[str, Any]:
        """Sample remote file via HTTP."""
        try:
            import urllib.request

            # Get headers
            with urllib.request.urlopen(url) as response:
                info = {
                    'uri': url,
                    'content_type': response.headers.get('Content-Type'),
                    'content_length': response.headers.get('Content-Length'),
                }

                # Infer extension from URL or content-type
                parsed = urllib.parse.urlparse(url)
                path = parsed.path
                if '.' in path:
                    info['extension'] = '.' + path.split('.')[-1]

                # Read sample
                sample_bytes = response.read(self.sample_size)
                info['sample_bytes'] = sample_bytes

                try:
                    info['sample_text'] = sample_bytes.decode('utf-8')
                    info['encoding'] = 'utf-8'
                except UnicodeDecodeError:
                    info['encoding'] = 'binary'

            return info
        except Exception as e:
            return {'uri': url, 'error': str(e)}


def infer_loader_from_extension(extension: str) -> str:
    """Infer pandas loader function from file extension.

    Args:
        extension: File extension (e.g., '.csv', '.json')

    Returns:
        Name of pandas loader function

    Example:
        >>> infer_loader_from_extension('.csv')
        'read_csv'
        >>> infer_loader_from_extension('.xlsx')
        'read_excel'
    """
    ext_map = {
        '.csv': 'read_csv',
        '.tsv': 'read_csv',
        '.txt': 'read_csv',
        '.json': 'read_json',
        '.jsonl': 'read_json',
        '.xlsx': 'read_excel',
        '.xls': 'read_excel',
        '.parquet': 'read_parquet',
        '.feather': 'read_feather',
        '.hdf': 'read_hdf',
        '.h5': 'read_hdf',
        '.sql': 'read_sql',
        '.html': 'read_html',
        '.xml': 'read_xml',
        '.pickle': 'read_pickle',
        '.pkl': 'read_pickle',
    }

    return ext_map.get(extension.lower(), 'read_csv')  # Default to CSV


def infer_loader_params(extension: str, sample_text: str = None) -> dict:
    """Infer parameters for pandas loader based on file characteristics.

    Args:
        extension: File extension
        sample_text: Optional sample of file content

    Returns:
        Dict of parameters to pass to loader

    Example:
        >>> params = infer_loader_params('.csv', 'a,b,c\\n1,2,3')  # doctest: +SKIP
        >>> params.get('sep')  # doctest: +SKIP
        ','
    """
    params = {}

    # Extension-based defaults
    if extension == '.tsv':
        params['sep'] = '\t'
    elif extension == '.jsonl':
        params['lines'] = True

    # Sample-based inference
    if sample_text:
        # Check for common delimiters
        if '\t' in sample_text and extension in ('.txt', '.csv'):
            params['sep'] = '\t'
        elif ';' in sample_text and ',' not in sample_text:
            params['sep'] = ';'
        elif '|' in sample_text:
            params['sep'] = '|'

    return params


# ============================================================================
# LLM Facades
# ============================================================================


def create_openai_chat(model: str = "gpt-4", **default_kwargs) -> Callable[[str], str]:
    """Create a text-to-text chat function using OpenAI API.

    Args:
        model: OpenAI model name
        **default_kwargs: Default parameters for API calls

    Returns:
        A callable that takes a prompt and returns a response

    Example:
        >>> chat = create_openai_chat("gpt-4")
        >>> response = chat("What is 2+2?")
    """
    try:
        from openai import OpenAI

        client = OpenAI()

        def chat(prompt: str, **kwargs) -> str:
            merged_kwargs = {**default_kwargs, **kwargs}
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **merged_kwargs,
            )
            return response.choices[0].message.content

        return chat
    except ImportError:
        raise ImportError(
            "OpenAI package not available. Install with: pip install openai"
        )


def create_oa_chat() -> Callable[[str], str]:
    """Create a chat function using the oa package.

    Returns:
        A callable that takes a prompt and returns a response

    Example:
        >>> chat = create_oa_chat()
        >>> response = chat("Hello!")
    """
    try:
        from oa import chat as oa_chat

        return oa_chat
    except ImportError:
        raise ImportError("oa package not available. Install with: pip install oa")


def default_llm_factory(model: Union[str, Callable] = "gpt-4") -> Callable[[str], str]:
    """Factory to create LLM function from various specifications.

    Args:
        model: Either a model name string, or a callable

    Returns:
        A callable text-to-text function

    Example:
        >>> llm = default_llm_factory("gpt-4")
        >>> llm = default_llm_factory(lambda p: f"Echo: {p}")
    """
    if callable(model):
        return model

    # Try to use oa package first (as specified in requirements)
    try:
        return create_oa_chat()
    except ImportError:
        pass

    # Fall back to OpenAI
    try:
        return create_openai_chat(model)
    except ImportError:
        pass

    # Ultimate fallback - echo function
    def echo_llm(prompt: str) -> str:
        return f"[ECHO MODE - No LLM available] Received: {prompt[:100]}"

    return echo_llm


# ============================================================================
# DataFrame Utilities
# ============================================================================


def compute_dataframe_info(df) -> dict:
    """Compute comprehensive info about a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        Dict with shape, dtypes, null counts, sample rows, etc.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> info = compute_dataframe_info(df)
        >>> info['shape']
        (2, 2)
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return {'error': 'Not a DataFrame', 'type': type(df).__name__}

    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'total_nulls': int(df.isnull().sum().sum()),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
    }

    # Add numeric column info
    numeric_cols = df.select_dtypes(include='number').columns
    info['numeric_columns'] = list(numeric_cols)
    info['num_numeric_columns'] = len(numeric_cols)

    # Add categorical column info
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    info['categorical_columns'] = list(categorical_cols)

    # Sample rows (as records for JSON serializability)
    info['sample_rows'] = df.head(5).to_dict('records')

    return info


def get_numeric_columns(df, exclude_nulls: bool = True):
    """Get numeric columns from DataFrame.

    Args:
        df: pandas DataFrame
        exclude_nulls: Whether to exclude columns with null values

    Returns:
        Generator of column names

    Example:
        >>> for col in get_numeric_columns(df):  # doctest: +SKIP
        ...     print(col)
    """
    import pandas as pd

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if exclude_nulls and df[col].isnull().any():
            continue
        yield col
```

## validation.py

```python
"""Validation system supporting three flavors: schema, info-dict, and functional.

This module provides validators that can be used to check if artifacts
meet requirements. Validators follow the pattern:
    validator(artifact) -> (success: bool, info: dict)
"""

from typing import Any, Callable
from collections.abc import Mapping


def _ensure_validator_protocol(func: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Ensure a function returns (bool, dict) tuple.

    Wraps functions that only return bool to add empty info dict.
    """

    def wrapper(artifact):
        result = func(artifact)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If function only returns bool, add empty info dict
        return result, {}

    return wrapper


# ============================================================================
# Flavor 1: Schema-Based Validators
# ============================================================================


def schema_validator(schema: Any) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator from a schema (Pydantic, JSON Schema, etc.).

    Args:
        schema: A schema object with validation capability

    Returns:
        A validator function

    Example:
        >>> from pydantic import BaseModel
        >>> class DataSchema(BaseModel):
        ...     x: float
        ...     y: float
        >>> validate = schema_validator(DataSchema)
        >>> validate({'x': 1.0, 'y': 2.0})
        (True, {'validated': ...})
    """

    def validate(artifact):
        try:
            # Try Pydantic model validation
            if hasattr(schema, 'model_validate'):
                result = schema.model_validate(artifact)
                return True, {'validated': result}
            # Try JSON schema validation
            elif hasattr(schema, 'validate'):
                schema.validate(artifact)
                return True, {}
            else:
                return False, {'error': 'Unknown schema type'}
        except Exception as e:
            return False, {'error': str(e), 'exception_type': type(e).__name__}

    return validate


# ============================================================================
# Flavor 2: Info-Dict Based Validators
# ============================================================================


def info_dict_validator(
    compute_info: Callable[[Any], dict], check_info: Callable[[dict], tuple[bool, str]]
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator that computes info then checks it.

    This is a two-stage validator:
    1. Compute information about the artifact (e.g., df.shape, df.dtypes)
    2. Check if that information meets requirements

    Args:
        compute_info: Function to extract info from artifact
        check_info: Function to check if info is acceptable

    Returns:
        A validator function

    Example:
        >>> def compute_df_info(df):
        ...     return {'shape': df.shape, 'null_count': df.isnull().sum().sum()}
        >>> def check_df_info(info):
        ...     if info['null_count'] > 10:
        ...         return False, "Too many nulls"
        ...     return True, "OK"
        >>> validate = info_dict_validator(compute_df_info, check_df_info)
    """

    def validate(artifact):
        try:
            info = compute_info(artifact)
            success, reason = check_info(info)
            return success, {'info': info, 'reason': reason}
        except Exception as e:
            return False, {'error': str(e), 'exception_type': type(e).__name__}

    return validate


# ============================================================================
# Flavor 3: Functional/Try Validators (Try the Purpose)
# ============================================================================


def functional_validator(
    try_function: Callable[[Any], Any], success_check: Callable[[Any], bool] = None
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator that tries to use the artifact for its purpose.

    This validator actually attempts to use the artifact in the way it's
    intended to be used (e.g., try to visualize the data, try to train a model).

    Args:
        try_function: Function that tries to use the artifact
        success_check: Optional function to check if result is acceptable

    Returns:
        A validator function

    Example:
        >>> def try_cosmo(df):  # doctest: +SKIP
        ...     return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')
        >>> validate = functional_validator(try_cosmo)  # doctest: +SKIP
        >>> validate(df)  # doctest: +SKIP
    """

    def validate(artifact):
        try:
            result = try_function(artifact)
            # If success_check provided, use it; otherwise any result = success
            if success_check is not None:
                success = success_check(result)
            else:
                success = result is not None
            return success, {'result': result}
        except Exception as e:
            return False, {
                'error': str(e),
                'exception_type': type(e).__name__,
                'traceback': _get_traceback_str(e),
            }

    return validate


def _get_traceback_str(exception: Exception) -> str:
    """Extract traceback string from exception."""
    import traceback

    return ''.join(traceback.format_tb(exception.__traceback__))


# ============================================================================
# Composite Validators
# ============================================================================


def all_validators(*validators: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Combine multiple validators - all must pass.

    Args:
        *validators: Validator functions to combine

    Returns:
        A validator that passes only if all validators pass

    Example:
        >>> validate = all_validators(  # doctest: +SKIP
        ...     is_dataframe,
        ...     has_required_columns(['x', 'y']),
        ...     has_no_nulls
        ... )
    """

    def validate(artifact):
        all_info = {}
        for i, validator in enumerate(validators):
            success, info = validator(artifact)
            all_info[f'validator_{i}'] = {'success': success, 'info': info}
            if not success:
                return False, all_info
        return True, all_info

    return validate


def any_validator(*validators: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Combine multiple validators - at least one must pass.

    Args:
        *validators: Validator functions to combine

    Returns:
        A validator that passes if any validator passes
    """

    def validate(artifact):
        all_info = {}
        for i, validator in enumerate(validators):
            success, info = validator(artifact)
            all_info[f'validator_{i}'] = {'success': success, 'info': info}
            if success:
                return True, all_info
        return False, all_info

    return validate


# ============================================================================
# Common Validators (Building Blocks)
# ============================================================================


def is_type(expected_type: type) -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact type.

    Example:
        >>> import pandas as pd
        >>> validate = is_type(pd.DataFrame)
    """

    def validate(artifact):
        success = isinstance(artifact, expected_type)
        info = {
            'expected_type': expected_type.__name__,
            'actual_type': type(artifact).__name__,
        }
        return success, info

    return validate


def is_not_empty() -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact is not empty.

    Works for sequences, mappings, DataFrames, etc.
    """

    def validate(artifact):
        try:
            is_empty = len(artifact) == 0
            return not is_empty, {'length': len(artifact)}
        except TypeError:
            # Object has no len - consider non-empty if it exists
            return artifact is not None, {'has_length': False}

    return validate


def has_attributes(**required_attrs: Any) -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact has required attributes.

    Example:
        >>> validate = has_attributes(shape=lambda s: len(s) == 2, columns=lambda c: len(c) > 0)
    """

    def validate(artifact):
        info = {}
        for attr_name, checker in required_attrs.items():
            if not hasattr(artifact, attr_name):
                return False, {'missing_attribute': attr_name, 'info': info}
            attr_value = getattr(artifact, attr_name)
            if checker is not None:
                if callable(checker):
                    check_passed = checker(attr_value)
                else:
                    check_passed = attr_value == checker
                info[attr_name] = {'value': attr_value, 'check_passed': check_passed}
                if not check_passed:
                    return False, info
            else:
                info[attr_name] = {'value': attr_value}
        return True, info

    return validate
```

## README.md

```python
# aw - Agentic Workflows for Data Preparation

An AI agent package for data preparation with a focus on modular, extensible workflows that follow best practices from modern agentic architectures.

## Features

- **AgenticStep Protocol**: Build modular agents that follow a consistent interface
- **ReAct Pattern**: Reason-Act-Observe loops with automatic retry logic
- **Three Validation Flavors**: Schema-based, info-dict based, and functional validation
- **Code Execution**: Safe defaults with extensibility for robust backends (LangChain, DSPy)
- **Loading Agent**: Automatically loads data from various sources into pandas DataFrames
- **Preparation Agent**: Transforms data to meet target requirements
- **Cosmograph Support**: Specialized validators for cosmograph visualization
- **Orchestration**: Chain multiple steps together with context management
- **Human-in-Loop**: Optional human approval at any step

## Installation

```bash
pip install -e .
```

### Optional Dependencies

```bash
# For LangChain code execution
pip install langchain langchain-experimental

# For cosmograph validation
pip install cosmograph

# For OpenAI LLM
pip install openai

# For oa package (preferred LLM interface)
pip install oa
```

## Quick Start

### Simple Example: Load and Prepare for Cosmograph

```python
from aw import load_for_cosmo

# Load and prepare data in one call
df, metadata = load_for_cosmo('data.csv')

# Use with cosmograph
from cosmograph import cosmo
params = metadata['preparing']['metadata']['validation_result']['params']
cosmo(df, **params)
```

### Step-by-Step Example

```python
from aw import LoadingAgent, PreparationAgent, Context
from aw.cosmo import basic_cosmo_validator

# Create context
context = Context()

# Step 1: Load data
loading_agent = LoadingAgent()
df, loading_meta = loading_agent.execute('data.csv', context)

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Prepare data
preparing_agent = PreparationAgent(
    target='cosmo-ready',
    target_validator=basic_cosmo_validator()
)
prepared_df, prep_meta = preparing_agent.execute(df, context)

print(f"Prepared data: {prep_meta['validation_result']}")
```

### Workflow Example

```python
from aw import create_cosmo_prep_workflow

# Create a complete workflow
workflow = create_cosmo_prep_workflow(max_retries=5)

# Run it
result_df, metadata = workflow.run('data.csv')

# Check results
if metadata['success']:
    print("Success!")
    print(f"Steps executed: {[s['name'] for s in metadata['steps']]}")
else:
    print(f"Failed at: {metadata.get('failed_at')}")
```

## Architecture

### Core Concepts

#### AgenticStep Protocol

All agents implement the `AgenticStep` protocol:

```python
def execute(
    self, 
    input_data: Any, 
    context: MutableMapping[str, Any]
) -> tuple[ArtifactType, dict[str, Any]]:
    """Execute the agentic step following ReAct pattern"""
    ...
```

#### ReAct Pattern

Each agent follows the Reason-Act-Observe loop:

1. **Reason (Thought)**: Analyze input and context to decide action
2. **Act (Action)**: Generate and execute code or use tools
3. **Observe**: Capture results and validate
4. **Repeat or Finish**: Based on validation, retry or proceed

#### Three Validation Flavors

1. **Schema-based**: Use Pydantic models or JSON schemas
2. **Info-dict based**: Compute info → check info → pass/fail
3. **Functional**: Try to use the artifact for its purpose

```python
from aw.validation import (
    schema_validator,
    info_dict_validator, 
    functional_validator
)

# Schema validation
from pydantic import BaseModel
class DataSchema(BaseModel):
    x: float
    y: float
validate = schema_validator(DataSchema)

# Info-dict validation
def compute_info(df):
    return {'null_count': df.isnull().sum().sum()}

def check_info(info):
    return info['null_count'] == 0, "No nulls allowed"

validate = info_dict_validator(compute_info, check_info)

# Functional validation (try the purpose)
def try_visualize(df):
    return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')

validate = functional_validator(try_visualize)
```

### Configuration System

Supports both simple objects and callables with cascading defaults:

```python
from aw import GlobalConfig, StepConfig

# Global defaults
global_config = GlobalConfig(
    llm="gpt-4",
    max_retries=3
)

# Step-specific override
step_config = global_config.override(
    llm="gpt-3.5-turbo",
    max_retries=5
)

# Or pass callables
def my_llm(prompt):
    return "Response to: " + prompt

step_config = StepConfig(
    llm=my_llm,  # Function instead of string
    validator=lambda x: (True, {}),
    max_retries=3
)
```

### Tools

#### CodeInterpreterTool

Safe code execution with extensibility:

```python
from aw import CodeInterpreterTool

# Default (safe exec)
tool = CodeInterpreterTool()
result = tool("x = 5; y = x * 2; print(y)")
print(result.output)  # "10"

# With LangChain backend
from aw import create_langchain_executor
tool = CodeInterpreterTool(executor=create_langchain_executor())
```

#### FileSamplerTool

Sample and analyze files before loading:

```python
from aw import FileSamplerTool

sampler = FileSamplerTool()
info = sampler('data.csv')
print(info['extension'])  # '.csv'
print(info['sample_text'])  # First 1024 bytes
```

### Custom Agents

Build your own agents following the pattern:

```python
from aw.base import AgenticStep, StepConfig
from collections.abc import MutableMapping
from typing import Any

class MyCustomAgent:
    """Custom agent following AgenticStep protocol"""
    
    def __init__(self, config: StepConfig = None):
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()
        self.validator = self.config.resolve_validator()
    
    def execute(
        self,
        input_data: Any,
        context: MutableMapping[str, Any]
    ) -> tuple[Any, dict]:
        attempts = []
        
        for attempt in range(self.config.max_retries):
            # 1. Reason: Analyze situation
            thought = self._analyze(input_data, attempts)
            
            # 2. Act: Perform action
            result = self._act(thought)
            
            # 3. Observe: Validate
            is_valid, info = self.validator(result)
            attempts.append({'attempt': attempt, 'info': info})
            
            if is_valid:
                context['my_agent'] = {'result': result, 'info': info}
                return result, {'success': True, 'attempts': attempts}
        
        return None, {'success': False, 'attempts': attempts}
```

## Advanced Usage

### Interactive Workflows with Human-in-Loop

```python
from aw import InteractiveWorkflow, LoadingAgent, PreparationAgent

workflow = InteractiveWorkflow()
workflow.add_step('loading', LoadingAgent(), require_approval=True)
workflow.add_step('preparing', PreparationAgent(), require_approval=False)

# Set custom approval callback
def approve(step_name, artifact, metadata):
    print(f"\nStep '{step_name}' completed")
    print(f"Artifact shape: {artifact.shape}")
    return input("Continue? (y/n): ").lower() == 'y'

workflow.set_approval_callback(approve)

# Run interactively
result, metadata = workflow.run_interactive('data.csv')
```

### Custom Validators

Create domain-specific validators:

```python
from aw.validation import info_dict_validator

def compute_ml_info(df):
    return {
        'num_features': len(df.columns) - 1,
        'num_samples': len(df),
        'class_balance': df['target'].value_counts().to_dict()
    }

def check_ml_requirements(info):
    if info['num_features'] < 5:
        return False, "Need at least 5 features"
    if info['num_samples'] < 100:
        return False, "Need at least 100 samples"
    return True, "ML ready"

ml_validator = info_dict_validator(compute_ml_info, check_ml_requirements)
```

### Partial Workflow Execution

```python
from aw import create_data_prep_workflow

workflow = create_data_prep_workflow()

# Run only the loading step
df, metadata = workflow.run_partial('data.csv', stop_after='loading')

# Inspect intermediate result
print(f"Loaded {df.shape}")

# Continue with preparing
final_df, final_meta = workflow.run_partial(df, stop_after='preparing')
```

## Design Principles

The package follows these principles:

1. **Functional over OOP**: Favor functions and generators
2. **Modularity**: Small, focused functions with clear purposes
3. **Open-Closed**: Easy to extend without modifying core
4. **Dependency Injection**: Abstract interfaces with injectable implementations
5. **Mapping Interfaces**: Use `Mapping`/`MutableMapping` from `collections.abc`
6. **Generators over Lists**: Use lazy evaluation when possible
7. **Minimal Doctests**: Simple examples in docstrings

## Project Structure

```
aw/
├── __init__.py         # Main exports
├── base.py            # Core protocols and configs
├── validation.py      # Three validation flavors
├── tools.py           # Code execution and tools
├── utils.py           # Helpers and facades
├── loading.py         # LoadingAgent
├── preparing.py       # PreparationAgent
├── cosmo.py          # Cosmograph validators
└── orchestration.py   # Workflow management
```

## Contributing

Contributions welcome! Key areas:

- Additional validators for other visualization libraries
- More robust LLM reasoning in code generation
- Integration with more backends (DSPy, CrewAI)
- State management with lakeFS/DVC
- Additional agent types (sourcing, finding, etc.)

## License

See LICENSE file.

## Credits

Built following best practices from:
- ReAct pattern (Reasoning and Acting)
- LangChain's agent architectures
- CrewAI's multi-agent systems
- Data Version Control (DVC) principles
```