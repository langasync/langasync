"""Comprehensive tests for get_parts_from_chain."""

import pytest

from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langasync.core.get_parts_from_chain import (
    get_parts_from_chain,
    ChainParts,
    _unwrap_to_model,
    _steps_to_runnable,
)
from langasync.core.exceptions import UnsupportedChainError
from .conftest import MockChatModel, MockLLM, AnotherMockChatModel, MockRetriever


class TestSingleRunnable:
    """Tests for single runnables that are not RunnableSequence."""

    def test_single_chat_model(self, chat_model: MockChatModel):
        """A single chat model returns model with passthrough pre/post."""
        parts = get_parts_from_chain(chat_model)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_single_llm(self, llm_model: MockLLM):
        """A single LLM returns model with passthrough pre/post."""
        parts = get_parts_from_chain(llm_model)

        assert parts.model is llm_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_single_prompt_template(self, chat_prompt: ChatPromptTemplate):
        """A single prompt (no model) returns None model."""
        parts = get_parts_from_chain(chat_prompt)

        assert parts.model is None
        assert parts.preprocessing is chat_prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_single_output_parser(self, str_parser: StrOutputParser):
        """A single parser (no model) returns None model."""
        parts = get_parts_from_chain(str_parser)

        assert parts.model is None
        assert parts.preprocessing is str_parser
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_single_passthrough(self):
        """RunnablePassthrough alone returns None model."""
        passthrough = RunnablePassthrough()
        parts = get_parts_from_chain(passthrough)

        assert parts.model is None
        assert parts.preprocessing is passthrough
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_single_lambda(self):
        """RunnableLambda alone returns None model."""
        lambda_runnable = RunnableLambda(lambda x: x.upper())
        parts = get_parts_from_chain(lambda_runnable)

        assert parts.model is None
        assert parts.preprocessing is lambda_runnable
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestStandardChainPatterns:
    """Tests for common chain patterns: prompt | model | parser."""

    def test_prompt_model_parser(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Standard chain: prompt | model | parser."""
        chain = chat_prompt | chat_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is str_parser

    def test_prompt_model(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Chain with no postprocessing: prompt | model."""
        chain = chat_prompt | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_model_parser(
        self,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Chain with no preprocessing: model | parser."""
        chain = chat_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert parts.postprocessing is str_parser

    def test_just_model_in_sequence(self, chat_model: MockChatModel):
        """A sequence containing only a model."""
        # Create a sequence with just the model
        chain = RunnablePassthrough() | chat_model
        # Remove the passthrough to get just model in sequence
        chain = chat_model | RunnablePassthrough()
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        # Preprocessing should be passthrough (no steps before model)
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        # Postprocessing should be the passthrough after model
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_no_model_in_sequence(
        self,
        chat_prompt: ChatPromptTemplate,
        str_parser: StrOutputParser,
    ):
        """Sequence with no model: prompt | parser."""
        chain = chat_prompt | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is None
        # Entire chain becomes preprocessing
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [chat_prompt, str_parser]
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_llm_in_chain(
        self,
        string_prompt: PromptTemplate,
        llm_model: MockLLM,
        str_parser: StrOutputParser,
    ):
        """Chain with LLM (not chat model)."""
        chain = string_prompt | llm_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is llm_model
        assert parts.preprocessing is string_prompt
        assert parts.postprocessing is str_parser


class TestMultipleSteps:
    """Tests for chains with multiple preprocessing or postprocessing steps."""

    def test_multiple_preprocessing_steps(self, chat_model: MockChatModel):
        """Multiple steps before model."""
        step1 = RunnableLambda(lambda x: {"input": x})
        step2 = RunnableLambda(lambda x: x["input"].upper())
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])

        chain = step1 | step2 | prompt | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [step1, step2, prompt]
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_multiple_postprocessing_steps(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Multiple steps after model."""
        step1 = StrOutputParser()
        step2 = RunnableLambda(lambda x: x.upper())
        step3 = RunnableLambda(lambda x: {"result": x})

        chain = chat_prompt | chat_model | step1 | step2 | step3
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert isinstance(parts.postprocessing, RunnableSequence)
        assert list(parts.postprocessing.steps) == [step1, step2, step3]

    def test_many_steps_both_sides(self, chat_model: MockChatModel):
        """Many steps on both sides of model."""
        pre1 = RunnableLambda(lambda x: x)
        pre2 = RunnableLambda(lambda x: x)
        pre3 = RunnableLambda(lambda x: x)
        post1 = RunnableLambda(lambda x: x)
        post2 = RunnableLambda(lambda x: x)
        post3 = RunnableLambda(lambda x: x)

        chain = pre1 | pre2 | pre3 | chat_model | post1 | post2 | post3
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [pre1, pre2, pre3]
        assert isinstance(parts.postprocessing, RunnableSequence)
        assert list(parts.postprocessing.steps) == [post1, post2, post3]

    def test_long_chain_no_model(self):
        """Long chain with no model."""
        steps = [RunnableLambda(lambda x: x) for _ in range(10)]
        chain = steps[0]
        for step in steps[1:]:
            chain = chain | step

        parts = get_parts_from_chain(chain)

        assert parts.model is None
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == steps
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestWrappedModels:
    """Tests for models wrapped with .bind(), .with_config(), etc."""

    def test_model_with_bind(self, chat_model: MockChatModel):
        """Model wrapped with .bind() is correctly unwrapped."""
        bound_model = chat_model.bind(temperature=0.5)
        parts = get_parts_from_chain(bound_model)

        assert parts.model is chat_model
        assert parts.model_bindings == {"temperature": 0.5}
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_model_with_config(self, chat_model: MockChatModel):
        """Model wrapped with .with_config() is correctly unwrapped."""
        configured_model = chat_model.with_config(tags=["test"])
        parts = get_parts_from_chain(configured_model)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_chain_with_bound_model(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Standard chain with bound model."""
        bound_model = chat_model.bind(stop=["\n"])
        chain = chat_prompt | bound_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is str_parser

    def test_nested_bindings(self, chat_model: MockChatModel):
        """Model with multiple nested bindings."""
        # model.bind().with_config().bind()
        wrapped = chat_model.bind(temperature=0.7)
        wrapped = wrapped.with_config(tags=["outer"])
        wrapped = wrapped.bind(max_tokens=100)

        parts = get_parts_from_chain(wrapped)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_deeply_nested_bindings_in_chain(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Chain with deeply nested bound model."""
        wrapped = chat_model.bind(a=1).bind(b=2).with_config(tags=["x"]).bind(c=3)
        chain = chat_prompt | wrapped
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.model_bindings == {"a": 1, "b": 2, "c": 3}
        assert parts.preprocessing is chat_prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_model_with_tools_binding(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Chain with model bound with tools."""
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        bound_model = chat_model.bind(tools=tools, tool_choice="auto")
        chain = chat_prompt | bound_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.model_bindings == {"tools": tools, "tool_choice": "auto"}
        assert parts.preprocessing is chat_prompt

    def test_model_with_tools_and_temperature(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Chain with model bound with both tools and temperature."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        bound_model = chat_model.bind(temperature=0.7).bind(tools=tools)
        chain = chat_prompt | bound_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.model_bindings == {"temperature": 0.7, "tools": tools}
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is str_parser


class TestMultipleModels:
    """Tests for chains with multiple models (should raise error)."""

    def test_two_models_raises_error(
        self,
        chat_model: MockChatModel,
        another_model: AnotherMockChatModel,
    ):
        """Chain with two models raises NotImplementedError."""
        chain = chat_model | another_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "multiple models" in str(exc_info.value).lower()

    def test_two_models_with_steps_between(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        another_model: AnotherMockChatModel,
        str_parser: StrOutputParser,
    ):
        """Chain with models separated by other steps raises error."""
        chain = chat_prompt | chat_model | str_parser | another_model

        with pytest.raises(UnsupportedChainError):
            get_parts_from_chain(chain)

    def test_three_models_raises_error(
        self,
        chat_model: MockChatModel,
        llm_model: MockLLM,
        another_model: AnotherMockChatModel,
    ):
        """Chain with three models raises error."""
        chain = chat_model | llm_model | another_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "3" in str(exc_info.value) or "multiple" in str(exc_info.value).lower()

    def test_two_bound_models_raises_error(
        self,
        chat_model: MockChatModel,
        another_model: AnotherMockChatModel,
    ):
        """Two bound models still detected and raise error."""
        bound1 = chat_model.bind(temperature=0.5)
        bound2 = another_model.with_config(tags=["test"])
        chain = bound1 | bound2

        with pytest.raises(UnsupportedChainError):
            get_parts_from_chain(chain)


class TestComplexRunnables:
    """Tests for complex runnables like RunnableParallel, RunnableLambda, etc."""

    def test_parallel_in_preprocessing(self, chat_model: MockChatModel):
        """RunnableParallel before model (no hidden models)."""
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x["input"]),
            b=RunnableLambda(lambda x: x["context"]),
        )
        chain = parallel | chat_model

        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is parallel
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_parallel_in_postprocessing(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """RunnableParallel after model."""
        parallel = RunnableParallel(
            original=RunnablePassthrough(),
            uppercase=RunnableLambda(lambda x: str(x).upper()),
        )
        chain = chat_prompt | chat_model | parallel

        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is parallel

    def test_lambda_with_passthrough(self, chat_model: MockChatModel):
        """Chain with lambdas and passthroughs."""
        pre_lambda = RunnableLambda(lambda x: {"input": x})
        pre_passthrough = RunnablePassthrough()
        post_lambda = RunnableLambda(lambda x: x.content)

        chain = pre_lambda | pre_passthrough | chat_model | post_lambda
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [pre_lambda, pre_passthrough]
        assert parts.postprocessing is post_lambda

    def test_passthrough_assign(self, chat_model: MockChatModel):
        """RunnablePassthrough.assign() in chain."""
        assign = RunnablePassthrough.assign(extra=RunnableLambda(lambda x: "extra"))
        chain = assign | chat_model

        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is assign
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_only_parallel(self):
        """RunnableParallel alone (no model)."""
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x),
            b=RunnableLambda(lambda x: x),
        )
        parts = get_parts_from_chain(parallel)

        assert parts.model is None
        assert parts.preprocessing is parallel
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_only_lambda(self):
        """RunnableLambda alone (no model)."""
        lambda_r = RunnableLambda(lambda x: x * 2)
        parts = get_parts_from_chain(lambda_r)

        assert parts.model is None
        assert parts.preprocessing is lambda_r
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_unwrap_to_model_direct_model(self, chat_model: MockChatModel):
        """_unwrap_to_model returns model directly with empty bindings."""
        model, bindings = _unwrap_to_model(chat_model)
        assert model is chat_model
        assert bindings == {}

    def test_unwrap_to_model_bound(self, chat_model: MockChatModel):
        """_unwrap_to_model unwraps bound model and returns bindings."""
        bound = chat_model.bind(temperature=0.5)
        model, bindings = _unwrap_to_model(bound)
        assert model is chat_model
        assert bindings == {"temperature": 0.5}

    def test_unwrap_to_model_configured(self, chat_model: MockChatModel):
        """_unwrap_to_model unwraps configured model."""
        configured = chat_model.with_config(tags=["test"])
        model, bindings = _unwrap_to_model(configured)
        assert model is chat_model

    def test_unwrap_to_model_nested(self, chat_model: MockChatModel):
        """_unwrap_to_model handles nested wrappings and merges bindings."""
        wrapped = chat_model.bind(a=1).with_config(tags=["x"]).bind(b=2)
        model, bindings = _unwrap_to_model(wrapped)
        assert model is chat_model
        assert bindings == {"a": 1, "b": 2}

    def test_unwrap_to_model_non_model(self, str_parser: StrOutputParser):
        """_unwrap_to_model returns None for non-model."""
        model, bindings = _unwrap_to_model(str_parser)
        assert model is None

    def test_unwrap_to_model_lambda(self):
        """_unwrap_to_model returns None for lambda."""
        lambda_r = RunnableLambda(lambda x: x)
        model, bindings = _unwrap_to_model(lambda_r)
        assert model is None

    def test_unwrap_to_model_with_tools(self, chat_model: MockChatModel):
        """_unwrap_to_model captures tools from .bind()."""
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        bound = chat_model.bind(tools=tools, tool_choice="auto")
        model, bindings = _unwrap_to_model(bound)
        assert model is chat_model
        assert bindings == {"tools": tools, "tool_choice": "auto"}

    def test_unwrap_to_model_with_tools_and_other_bindings(self, chat_model: MockChatModel):
        """_unwrap_to_model captures tools alongside other bindings."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        bound = chat_model.bind(temperature=0.7).bind(tools=tools)
        model, bindings = _unwrap_to_model(bound)
        assert model is chat_model
        assert bindings == {"temperature": 0.7, "tools": tools}

    def test_steps_to_runnable_empty(self):
        """_steps_to_runnable returns passthrough for empty list."""
        result = _steps_to_runnable([])
        assert isinstance(result, RunnablePassthrough)

    def test_steps_to_runnable_single(self, str_parser: StrOutputParser):
        """_steps_to_runnable returns single step directly."""
        result = _steps_to_runnable([str_parser])
        assert result is str_parser

    def test_steps_to_runnable_multiple(self, str_parser: StrOutputParser):
        """_steps_to_runnable chains multiple steps."""
        lambda1 = RunnableLambda(lambda x: x)
        lambda2 = RunnableLambda(lambda x: x)
        result = _steps_to_runnable([lambda1, str_parser, lambda2])
        assert isinstance(result, RunnableSequence)
        assert list(result.steps) == [lambda1, str_parser, lambda2]


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_model_at_start(
        self,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Model at the very start of chain."""
        chain = chat_model | str_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert parts.postprocessing is str_parser

    def test_model_at_end(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Model at the very end of chain."""
        chain = chat_prompt | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_model_in_middle_of_long_chain(self, chat_model: MockChatModel):
        """Model in middle of very long chain."""
        pre_steps = [RunnableLambda(lambda x: x) for _ in range(5)]
        post_steps = [RunnableLambda(lambda x: x) for _ in range(5)]

        chain = pre_steps[0]
        for step in pre_steps[1:]:
            chain = chain | step
        chain = chain | chat_model
        for step in post_steps:
            chain = chain | step

        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == pre_steps
        assert isinstance(parts.postprocessing, RunnableSequence)
        assert list(parts.postprocessing.steps) == post_steps

    def test_preserves_chain_structure(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """Verifies preprocessing and postprocessing preserve original runnables."""
        chain = chat_prompt | chat_model | str_parser
        parts = get_parts_from_chain(chain)

        # The actual runnable objects should be preserved (not copies)
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is str_parser

    def test_chain_parts_dataclass_fields(
        self,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
        str_parser: StrOutputParser,
    ):
        """ChainParts dataclass has expected fields."""
        chain = chat_prompt | chat_model | str_parser
        parts = get_parts_from_chain(chain)

        assert hasattr(parts, "preprocessing")
        assert hasattr(parts, "model")
        assert hasattr(parts, "postprocessing")


class TestLLMvsChatModel:
    """Tests to ensure both LLM and ChatModel types work correctly."""

    def test_llm_detected(self, llm_model: MockLLM):
        """BaseLLM is detected as a model."""
        parts = get_parts_from_chain(llm_model)
        assert parts.model is llm_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_chat_model_detected(self, chat_model: MockChatModel):
        """BaseChatModel is detected as a model."""
        parts = get_parts_from_chain(chat_model)
        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_llm_in_sequence(
        self,
        string_prompt: PromptTemplate,
        llm_model: MockLLM,
    ):
        """LLM in sequence is detected."""
        chain = string_prompt | llm_model
        parts = get_parts_from_chain(chain)

        assert parts.model is llm_model
        assert parts.preprocessing is string_prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_bound_llm(self, llm_model: MockLLM):
        """Bound LLM is unwrapped correctly."""
        bound = llm_model.bind(temperature=0.5)
        parts = get_parts_from_chain(bound)

        assert parts.model is llm_model
        assert isinstance(parts.preprocessing, RunnablePassthrough)
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestRealisticScenarios:
    """Tests that mirror real-world usage patterns."""

    def test_rag_chain_pattern(self, chat_model: MockChatModel):
        """RAG-style chain: context formatting | prompt | model | parser."""
        context_formatter = RunnableLambda(lambda x: {"context": x["docs"], "question": x["q"]})
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Context: {context}"),
                ("user", "{question}"),
            ]
        )
        parser = StrOutputParser()

        chain = context_formatter | prompt | chat_model | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [context_formatter, prompt]
        assert parts.postprocessing is parser

    def test_extraction_chain_pattern(self, chat_model: MockChatModel):
        """Extraction chain with structured output parsing."""
        prompt = ChatPromptTemplate.from_messages([("user", "Extract: {text}")])
        # Simulating a json output parser
        json_parser = RunnableLambda(lambda x: {"extracted": str(x)})

        chain = prompt | chat_model | json_parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is prompt
        assert parts.postprocessing is json_parser

    def test_summarization_chain(self, chat_model: MockChatModel):
        """Summarization chain pattern."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize the following text."),
                ("user", "{text}"),
            ]
        )
        parser = StrOutputParser()
        chain = prompt | chat_model.bind(max_tokens=500) | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is prompt
        assert parts.postprocessing is parser

    def test_chat_with_history_pattern(self, chat_model: MockChatModel):
        """Chat chain with history formatting."""
        history_formatter = RunnableLambda(
            lambda x: {
                "history": "\n".join(x.get("history", [])),
                "input": x["input"],
            }
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Previous conversation:\n{history}"),
                ("user", "{input}"),
            ]
        )

        chain = history_formatter | prompt | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.preprocessing, RunnableSequence)
        assert list(parts.preprocessing.steps) == [history_formatter, prompt]
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestUnsupportedComponents:
    """Tests for components that should raise NotImplementedError."""

    def test_retriever_alone_raises_error(self, mock_retriever):
        """A retriever alone raises NotImplementedError."""
        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(mock_retriever)

        assert "retriever" in str(exc_info.value).lower()

    def test_retriever_in_preprocessing_raises_error(
        self,
        mock_retriever,
        chat_model: MockChatModel,
    ):
        """Retriever before model raises NotImplementedError."""
        chain = mock_retriever | chat_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "retriever" in str(exc_info.value).lower()

    def test_retriever_in_chain_with_prompt_raises_error(
        self,
        mock_retriever,
        chat_prompt: ChatPromptTemplate,
        chat_model: MockChatModel,
    ):
        """Retriever in RAG-style chain raises NotImplementedError."""
        format_docs = RunnableLambda(lambda docs: "\n".join(d.page_content for d in docs))
        chain = mock_retriever | format_docs | chat_prompt | chat_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "retriever" in str(exc_info.value).lower()

    def test_bound_retriever_raises_error(self, mock_retriever):
        """Bound retriever still raises NotImplementedError."""
        bound_retriever = mock_retriever.with_config(tags=["test"])

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(bound_retriever)

        assert "retriever" in str(exc_info.value).lower()


class TestOtherOutputParsers:
    """Tests for various output parsers."""

    def test_json_output_parser(self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel):
        """JsonOutputParser in chain works correctly."""
        from langchain_core.output_parsers import JsonOutputParser

        parser = JsonOutputParser()
        chain = chat_prompt | chat_model | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is parser

    def test_pydantic_output_parser(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """PydanticOutputParser in chain works correctly."""
        from langchain_core.output_parsers import PydanticOutputParser
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        parser = PydanticOutputParser(pydantic_object=TestSchema)
        chain = chat_prompt | chat_model | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is parser

    def test_comma_separated_list_parser(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """CommaSeparatedListOutputParser in chain works correctly."""
        from langchain_core.output_parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        chain = chat_prompt | chat_model | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is parser


class TestOtherRunnables:
    """Tests for other runnable types."""

    def test_runnable_pick_in_preprocessing(self, chat_model: MockChatModel):
        """RunnablePick before model works correctly."""
        from langchain_core.runnables import RunnablePick

        pick = RunnablePick(keys=["input"])
        chain = pick | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is pick
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_runnable_pick_in_postprocessing(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """RunnablePick after model works correctly."""
        from langchain_core.runnables import RunnablePick

        pick = RunnablePick(keys=["content"])
        chain = chat_prompt | chat_model | pick
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is pick

    def test_itemgetter_in_preprocessing(self, chat_model: MockChatModel):
        """itemgetter (via RunnableLambda) before model works correctly."""
        from operator import itemgetter

        getter = RunnableLambda(itemgetter("input"))
        chain = getter | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is getter
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_runnable_generator_in_preprocessing(self, chat_model: MockChatModel):
        """RunnableGenerator before model works correctly."""
        from langchain_core.runnables import RunnableGenerator

        def gen(input):
            yield input

        generator = RunnableGenerator(gen)
        chain = generator | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is generator
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_runnable_generator_in_postprocessing(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """RunnableGenerator after model works correctly."""
        from langchain_core.runnables import RunnableGenerator

        def gen(input):
            yield str(input)

        generator = RunnableGenerator(gen)
        chain = chat_prompt | chat_model | generator
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is generator

    def test_xml_output_parser(self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel):
        """XMLOutputParser in chain works correctly."""
        from langchain_core.output_parsers import XMLOutputParser

        parser = XMLOutputParser()
        chain = chat_prompt | chat_model | parser
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is chat_prompt
        assert parts.postprocessing is parser


class TestPromptComponents:
    """Tests for various prompt template types."""

    def test_messages_placeholder_in_prompt(self, chat_model: MockChatModel):
        """ChatPromptTemplate with MessagesPlaceholder works correctly."""
        from langchain_core.prompts import MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful."),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{input}"),
            ]
        )
        chain = prompt | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is prompt
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_few_shot_prompt_template(self, chat_model: MockChatModel):
        """FewShotPromptTemplate in chain works correctly."""
        from langchain_core.prompts import FewShotPromptTemplate

        examples = [
            {"input": "hello", "output": "hi"},
            {"input": "bye", "output": "goodbye"},
        ]
        example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")
        few_shot = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Translate the following:",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"],
        )
        chain = few_shot | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is few_shot
        assert isinstance(parts.postprocessing, RunnablePassthrough)


class TestRejectedRunnables:
    """Tests for runnables that should raise UnsupportedChainError."""

    def test_runnable_branch_raises_error(self, chat_model: MockChatModel):
        """RunnableBranch alone raises UnsupportedChainError."""
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: x.get("type") == "a", RunnableLambda(lambda x: "branch a")),
            RunnableLambda(lambda x: "default"),
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(branch)

        assert "branch" in str(exc_info.value).lower()

    def test_runnable_branch_in_chain_raises_error(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """RunnableBranch in chain raises UnsupportedChainError."""
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: len(x) > 10, chat_model),
            RunnableLambda(lambda x: "short"),
        )
        chain = chat_prompt | branch

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "branch" in str(exc_info.value).lower()

    def test_bound_runnable_branch_raises_error(self):
        """Bound RunnableBranch still raises UnsupportedChainError."""
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: True, RunnableLambda(lambda x: x)),
            RunnableLambda(lambda x: x),
        )
        bound = branch.with_config(tags=["test"])

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(bound)

        assert "branch" in str(exc_info.value).lower()

    def test_runnable_each_raises_error(self, chat_model: MockChatModel):
        """RunnableEach raises UnsupportedChainError."""
        from langchain_core.runnables.base import RunnableEach

        each = RunnableEach(bound=RunnableLambda(lambda x: x.upper()))

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(each)

        assert "each" in str(exc_info.value).lower()

    def test_runnable_each_in_chain_raises_error(
        self, chat_prompt: ChatPromptTemplate, chat_model: MockChatModel
    ):
        """RunnableEach in chain raises UnsupportedChainError."""
        from langchain_core.runnables.base import RunnableEach

        each = RunnableEach(bound=RunnableLambda(lambda x: x))
        chain = each | chat_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "each" in str(exc_info.value).lower()

    def test_bound_runnable_each_raises_error(self):
        """Bound RunnableEach still raises UnsupportedChainError."""
        from langchain_core.runnables.base import RunnableEach

        each = RunnableEach(bound=RunnableLambda(lambda x: x))
        bound = each.with_config(tags=["test"])

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(bound)

        assert "each" in str(exc_info.value).lower()


class TestHiddenModels:
    """Tests for models hidden inside container runnables."""

    def test_model_hidden_in_parallel_raises_error(
        self, chat_model: MockChatModel, another_model: AnotherMockChatModel
    ):
        """Model inside RunnableParallel raises UnsupportedChainError."""
        parallel = RunnableParallel(
            branch_a=chat_model,
            branch_b=RunnableLambda(lambda x: x),
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "hidden" in str(exc_info.value).lower()

    def test_model_hidden_in_parallel_with_top_level_model(
        self, chat_model: MockChatModel, another_model: AnotherMockChatModel
    ):
        """Model in RunnableParallel plus top-level model raises error."""
        parallel = RunnableParallel(
            branch_a=another_model,
            branch_b=RunnableLambda(lambda x: x),
        )
        chain = parallel | chat_model

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "hidden" in str(exc_info.value).lower()

    def test_multiple_models_hidden_in_parallel(
        self, chat_model: MockChatModel, another_model: AnotherMockChatModel
    ):
        """Multiple models inside RunnableParallel raises error."""
        parallel = RunnableParallel(
            branch_a=chat_model,
            branch_b=another_model,
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "hidden" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)

    def test_bound_model_hidden_in_parallel(self, chat_model: MockChatModel):
        """Bound model inside RunnableParallel is still detected."""
        bound_model = chat_model.bind(temperature=0.5)
        parallel = RunnableParallel(
            branch_a=bound_model,
            branch_b=RunnableLambda(lambda x: x),
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "hidden" in str(exc_info.value).lower()

    def test_model_in_nested_parallel(
        self, chat_model: MockChatModel, chat_prompt: ChatPromptTemplate
    ):
        """Model in nested RunnableParallel raises error."""
        inner_parallel = RunnableParallel(
            inner_a=chat_model,
            inner_b=RunnableLambda(lambda x: x),
        )
        outer_parallel = RunnableParallel(
            outer_a=inner_parallel,
            outer_b=RunnableLambda(lambda x: x),
        )
        chain = chat_prompt | outer_parallel

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "hidden" in str(exc_info.value).lower()

    def test_parallel_without_models_still_works(self, chat_model: MockChatModel):
        """RunnableParallel without hidden models works fine."""
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x["input"]),
            b=RunnableLambda(lambda x: x["context"]),
        )
        chain = parallel | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert parts.preprocessing is parallel
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_model_hidden_in_assign_raises_error(self, chat_model: MockChatModel):
        """Model inside RunnablePassthrough.assign() raises UnsupportedChainError."""
        chain = RunnablePassthrough.assign(result=chat_model)

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(chain)

        assert "hidden" in str(exc_info.value).lower()

    def test_assign_without_model_works(self, chat_model: MockChatModel):
        """RunnablePassthrough.assign() without models works fine."""
        chain = RunnablePassthrough.assign(foo=RunnableLambda(lambda x: x["input"])) | chat_model
        parts = get_parts_from_chain(chain)

        assert parts.model is chat_model
        assert isinstance(parts.postprocessing, RunnablePassthrough)

    def test_nested_branch_in_parallel_raises_error(self):
        """RunnableBranch nested inside RunnableParallel raises error."""
        from langchain_core.runnables.branch import RunnableBranch

        branch = RunnableBranch(
            (lambda x: x > 0, RunnableLambda(lambda x: "positive")),
            RunnableLambda(lambda x: "negative"),
        )
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x),
            b=branch,
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "branch" in str(exc_info.value).lower()

    def test_nested_each_in_parallel_raises_error(self):
        """RunnableEach nested inside RunnableParallel raises error."""
        from langchain_core.runnables.base import RunnableEach

        each = RunnableEach(bound=RunnableLambda(lambda x: x))
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x),
            b=each,
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "each" in str(exc_info.value).lower()

    def test_nested_retriever_in_parallel_raises_error(self, mock_retriever: MockRetriever):
        """Retriever nested inside RunnableParallel raises error."""
        parallel = RunnableParallel(
            a=RunnableLambda(lambda x: x),
            b=mock_retriever,
        )

        with pytest.raises(UnsupportedChainError) as exc_info:
            get_parts_from_chain(parallel)

        assert "retriever" in str(exc_info.value).lower()
