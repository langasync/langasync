"""Extract model and pre/post processing parts from LangChain chains."""

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableSequence, RunnablePassthrough
from langchain_core.runnables.base import RunnableBindingBase, RunnableEachBase
from langchain_core.runnables.branch import RunnableBranch

from langasync.core.exceptions import UnsupportedChainError


@dataclass
class ChainParts:
    """The decomposed parts of a chain."""

    preprocessing: Runnable
    model: Optional[BaseLanguageModel]
    postprocessing: Runnable


def get_parts_from_chain(chain: Runnable) -> ChainParts:
    """Decompose a chain into preprocessing, model, and postprocessing parts.

    For a chain like: prompt | model | parser
    Returns:
        - preprocessing: prompt
        - model: model (unwrapped)
        - postprocessing: parser

    Args:
        chain: A LangChain Runnable (typically a RunnableSequence)

    Returns:
        ChainParts with the three components

    Raises:
        UnsupportedChainError: If chain has multiple models, retrievers, RunnableBranch, or RunnableEach
    """
    # Get steps from a RunnableSequence, or treat single runnable as one step
    if isinstance(chain, RunnableSequence):
        steps = list(chain.steps)
    else:
        steps = [chain]

    # Check for unsupported components
    for step in steps:
        if _is_retriever(step):
            raise UnsupportedChainError(
                "Chains containing retrievers are not supported. "
                "Retrievers should be executed separately before batch submission."
            )
        if _is_branching_runnable(step):
            raise UnsupportedChainError(
                "Chains containing RunnableBranch are not supported. "
                "Branching logic may contain hidden models that cannot be analyzed."
            )
        if _is_each_runnable(step):
            raise UnsupportedChainError(
                "Chains containing RunnableEach are not supported. "
                "RunnableEach may wrap models that cannot be analyzed."
            )

    # Find all model indices
    model_indices: List[int] = []
    for i, step in enumerate(steps):
        if _unwrap_to_model(step) is not None:
            model_indices.append(i)

    # Validate: at most one model
    if len(model_indices) > 1:
        raise UnsupportedChainError(
            f"Chains with multiple models are not supported. Found {len(model_indices)} models."
        )

    # No model case
    if len(model_indices) == 0:
        return ChainParts(
            preprocessing=chain,
            model=None,
            postprocessing=RunnablePassthrough(),
        )

    # Single model case
    model_idx = model_indices[0]
    model = _unwrap_to_model(steps[model_idx])

    # Split into pre and post
    pre_steps = steps[:model_idx]
    post_steps = steps[model_idx + 1 :]

    preprocessing = _steps_to_runnable(pre_steps)
    postprocessing = _steps_to_runnable(post_steps)

    return ChainParts(
        preprocessing=preprocessing,
        model=model,
        postprocessing=postprocessing,
    )


def _steps_to_runnable(steps: List[Runnable]) -> Runnable:
    """Convert a list of steps back into a Runnable."""
    if len(steps) == 0:
        return RunnablePassthrough()
    elif len(steps) == 1:
        return steps[0]
    else:
        # Chain them together
        result = steps[0]
        for step in steps[1:]:
            result = result | step
        return result


def _unwrap_to_model(runnable) -> Optional[BaseLanguageModel]:
    """Unwrap a runnable to get the underlying BaseLanguageModel, if any.

    Handles:
    - Direct BaseLanguageModel
    - RunnableBindingBase (.bind(), .with_config(), .with_structured_output(), etc.)

    Args:
        runnable: Any LangChain Runnable

    Returns:
        The underlying BaseLanguageModel, or None if not a model
    """
    # Direct model
    if isinstance(runnable, BaseLanguageModel):
        return runnable

    # Unwrap RunnableBindingBase (.bind(), .with_config(), etc.)
    if isinstance(runnable, RunnableBindingBase):
        return _unwrap_to_model(runnable.bound)

    # Not a model
    return None


def _is_retriever(runnable) -> bool:
    """Check if a runnable is a retriever.

    Args:
        runnable: Any LangChain Runnable

    Returns:
        True if the runnable is a BaseRetriever, False otherwise
    """
    # Direct retriever
    if isinstance(runnable, BaseRetriever):
        return True

    # Unwrap RunnableBindingBase to check if it wraps a retriever
    if isinstance(runnable, RunnableBindingBase):
        return _is_retriever(runnable.bound)

    return False


def _is_branching_runnable(runnable) -> bool:
    """Check if a runnable is a RunnableBranch.

    Args:
        runnable: Any LangChain Runnable

    Returns:
        True if the runnable is a RunnableBranch, False otherwise
    """
    if isinstance(runnable, RunnableBranch):
        return True

    if isinstance(runnable, RunnableBindingBase):
        return _is_branching_runnable(runnable.bound)

    return False


def _is_each_runnable(runnable) -> bool:
    """Check if a runnable is a RunnableEach.

    Args:
        runnable: Any LangChain Runnable

    Returns:
        True if the runnable is a RunnableEachBase, False otherwise
    """
    if isinstance(runnable, RunnableEachBase):
        return True

    if isinstance(runnable, RunnableBindingBase):
        return _is_each_runnable(runnable.bound)

    return False
