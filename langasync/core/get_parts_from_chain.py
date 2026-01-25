"""Extract model and pre/post processing parts from LangChain chains."""

from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.runnables.base import RunnableBindingBase, RunnableEachBase
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign

from langasync.core.exceptions import UnsupportedChainError


@dataclass
class ChainParts:
    """The decomposed parts of a chain."""

    preprocessing: Runnable
    model: BaseLanguageModel | None
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
        UnsupportedChainError: If chain has multiple models, hidden models in containers,
            retrievers, RunnableBranch, or RunnableEach
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

    # Find model and scan for hidden models in a single pass
    single_model_index: int | None = None
    for i, step in enumerate(steps):
        _model = _unwrap_to_model(step)

        model_set_in_previous_step = single_model_index is not None
        if _model is not None and model_set_in_previous_step:
            raise UnsupportedChainError("Chains with multiple models are not supported.")
        elif _model is not None:
            single_model_index = i
        else:
            # Not a top-level model, check for hidden models inside containers
            hidden = _find_hidden_models(step)
            if hidden:
                raise UnsupportedChainError(
                    f"Found {len(hidden)} model(s) hidden inside a container. "
                    "Models must be at the top level of the chain, not nested inside "
                    "RunnableParallel or similar containers."
                )

    # No model case
    if single_model_index is None:
        return ChainParts(
            preprocessing=chain,
            model=None,
            postprocessing=RunnablePassthrough(),
        )

    # Split into pre and post
    pre_steps = steps[:single_model_index]
    post_steps = steps[single_model_index + 1 :]

    preprocessing = _steps_to_runnable(pre_steps)
    postprocessing = _steps_to_runnable(post_steps)

    return ChainParts(
        preprocessing=preprocessing,
        model=_unwrap_to_model(steps[single_model_index]),
        postprocessing=postprocessing,
    )


def _steps_to_runnable(steps: list[Runnable]) -> Runnable:
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


def _unwrap_to_model(runnable) -> BaseLanguageModel | None:
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


def _find_hidden_models(runnable) -> list[BaseLanguageModel]:
    """Recursively find models hidden inside container runnables.

    Inspects inside RunnableParallel, RunnableAssign, and similar containers
    to find models that would be missed by top-level scanning.

    Also raises UnsupportedChainError if unsupported components (RunnableBranch,
    RunnableEach, retrievers) are found nested inside containers.

    Args:
        runnable: Any LangChain Runnable

    Returns:
        List of models found inside the runnable (empty if none)

    Raises:
        UnsupportedChainError: If unsupported components are found nested
    """
    models: list[BaseLanguageModel] = []

    # Unwrap bindings first
    if isinstance(runnable, RunnableBindingBase):
        return _find_hidden_models(runnable.bound)

    # Check for unsupported types nested inside containers
    if isinstance(runnable, RunnableBranch):
        raise UnsupportedChainError(
            "Chains containing RunnableBranch are not supported. "
            "Branching logic may contain hidden models that cannot be analyzed."
        )
    if isinstance(runnable, RunnableEachBase):
        raise UnsupportedChainError(
            "Chains containing RunnableEach are not supported. "
            "RunnableEach may wrap models that cannot be analyzed."
        )
    if isinstance(runnable, BaseRetriever):
        raise UnsupportedChainError(
            "Chains containing retrievers are not supported. "
            "Retrievers should be executed separately before batch submission."
        )

    # Check inside RunnableAssign (from .assign())
    if isinstance(runnable, RunnableAssign):
        # RunnableAssign.mapper is a RunnableParallel
        return _find_hidden_models(runnable.mapper)

    # Check inside RunnableParallel branches
    if isinstance(runnable, RunnableParallel):
        # RunnableParallel.steps__ is a dict of key -> Runnable
        for branch in runnable.steps__.values():
            # Check if branch itself is a model
            model = _unwrap_to_model(branch)
            if model:
                models.append(model)
            else:
                # Recurse into the branch
                models.extend(_find_hidden_models(branch))
        return models

    # Check inside nested RunnableSequence
    if isinstance(runnable, RunnableSequence):
        for step in runnable.steps:
            model = _unwrap_to_model(step)
            if model:
                models.append(model)
            else:
                models.extend(_find_hidden_models(step))
        return models

    return models
