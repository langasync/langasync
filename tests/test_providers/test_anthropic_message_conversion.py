"""Test custom_convert_to_anthropic_messages behavior on different input types.

This tests our adapter function that wraps langchain-anthropic's _format_messages.
If these tests fail after a langchain-anthropic upgrade, the underlying function
signature or behavior has changed and our adapter needs updating.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langasync.providers.anthropic import custom_convert_to_anthropic_messages


class TestConvertToAnthropicMessages:
    """Test custom_convert_to_anthropic_messages with different LanguageModelInput types."""

    def test_string_returns_user_message(self):
        system, messages = custom_convert_to_anthropic_messages("Hello world")
        assert system is None
        assert messages == [{"role": "user", "content": "Hello world"}]

    def test_human_message_returns_user_role(self):
        system, messages = custom_convert_to_anthropic_messages([HumanMessage("Hello")])
        assert system is None
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_single_human_message_without_list(self):
        system, messages = custom_convert_to_anthropic_messages(HumanMessage("Hello"))
        assert system is None
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_single_system_message_without_list(self):
        system, messages = custom_convert_to_anthropic_messages(SystemMessage("You are helpful"))
        assert system == "You are helpful"
        assert messages == []

    def test_single_ai_message_without_list(self):
        system, messages = custom_convert_to_anthropic_messages(AIMessage("Hi there"))
        assert system is None
        assert messages == [{"role": "assistant", "content": "Hi there"}]

    def test_system_message_extracted_separately(self):
        system, messages = custom_convert_to_anthropic_messages(
            [SystemMessage("You are helpful"), HumanMessage("Hello")]
        )
        assert system == "You are helpful"
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_ai_message_returns_assistant_role(self):
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("Hello"), AIMessage("Hi there")]
        )
        assert system is None
        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_list_of_messages_returns_list(self):
        system, messages = custom_convert_to_anthropic_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage("Hello"),
                AIMessage("Hi!"),
            ]
        )
        assert system == "You are helpful"
        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_conversation_with_system(self):
        system, messages = custom_convert_to_anthropic_messages(
            [
                SystemMessage("Be concise"),
                HumanMessage("What is 2+2?"),
                AIMessage("4"),
                HumanMessage("And 3+3?"),
            ]
        )
        assert system == "Be concise"
        assert messages == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

    def test_system_message_only(self):
        """System message alone (edge case - needs at least one user message in practice)."""
        system, messages = custom_convert_to_anthropic_messages([SystemMessage("You are helpful")])
        assert system == "You are helpful"
        assert messages == []

    def test_multiple_human_messages(self):
        """Multiple consecutive human messages."""
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("Hello"), HumanMessage("How are you?")]
        )
        assert system is None
        # Anthropic merges consecutive same-role messages
        assert len(messages) >= 1
        assert messages[0]["role"] == "user"

    def test_ai_message_only(self):
        """Single AI message."""
        system, messages = custom_convert_to_anthropic_messages([AIMessage("Hi there")])
        assert system is None
        assert messages == [{"role": "assistant", "content": "Hi there"}]

    def test_empty_list(self):
        """Empty message list."""
        system, messages = custom_convert_to_anthropic_messages([])
        assert system is None
        assert messages == []

    def test_returns_tuple(self):
        """Verify return type is tuple of (system, messages)."""
        result = custom_convert_to_anthropic_messages([HumanMessage("Hello")])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_image_url_content(self):
        """HumanMessage with image URL gets converted to Anthropic image source format."""
        system, messages = custom_convert_to_anthropic_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "url": "https://example.com/cat.jpg"},
                    ]
                ),
            ]
        )
        assert system == "You are helpful"
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
                    },
                ],
            },
        ]

    def test_base64_image_content(self):
        """HumanMessage with base64 image gets converted to Anthropic base64 source format."""
        system, messages = custom_convert_to_anthropic_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "base64": "iVBOR...", "mime_type": "image/png"},
                    ]
                ),
            ]
        )
        assert system is None
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBOR...",
                        },
                    },
                ],
            },
        ]

    def test_ai_message_with_tool_calls(self):
        """AI message with tool_calls gets converted to tool_use blocks."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("What is the weather?"), ai_with_tools]
        )
        assert system is None
        assert messages == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "id": "call_123",
                    }
                ],
            },
        ]

    def test_tool_message_becomes_tool_result(self):
        """ToolMessage gets converted to tool_result block."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        tool_response = ToolMessage(content="72 degrees", tool_call_id="call_123")
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("What is the weather?"), ai_with_tools, tool_response]
        )
        assert system is None
        assert messages == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "id": "call_123",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "72 degrees",
                        "tool_use_id": "call_123",
                        "is_error": False,
                    }
                ],
            },
        ]

    def test_ai_message_with_text_and_tool_calls(self):
        """AI message with both text content and tool_calls includes both."""
        ai_with_tools = AIMessage(
            content="I'll check the weather.",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("What is the weather?"), ai_with_tools]
        )
        assert system is None
        assert messages == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll check the weather."},
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "id": "call_123",
                    },
                ],
            },
        ]

    def test_multiple_tool_calls_in_ai_message(self):
        """AI message with multiple tool calls."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "NYC"}, "id": "call_1"},
                {"name": "get_time", "args": {"timezone": "EST"}, "id": "call_2"},
            ],
        )
        system, messages = custom_convert_to_anthropic_messages(
            [HumanMessage("Weather and time?"), ai_with_tools]
        )
        assert system is None
        assert messages == [
            {"role": "user", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "id": "call_1",
                    },
                    {
                        "type": "tool_use",
                        "name": "get_time",
                        "input": {"timezone": "EST"},
                        "id": "call_2",
                    },
                ],
            },
        ]

    def test_data_uri_image(self):
        """Data URI image gets converted to base64 source format."""
        system, messages = custom_convert_to_anthropic_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "url": "data:image/png;base64,iVBOR"},
                    ]
                ),
            ]
        )
        assert system is None
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBOR",
                        },
                    },
                ],
            },
        ]

    def test_openai_image_url_format(self):
        """OpenAI-style image_url content is converted to Anthropic format."""
        system, messages = custom_convert_to_anthropic_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    ]
                ),
            ]
        )
        assert system is None
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/img.jpg"},
                    },
                ],
            },
        ]

    def test_prompt_value_input(self):
        """PromptValue from ChatPromptTemplate is handled correctly."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful"),
                ("user", "Hello {name}"),
            ]
        )
        prompt_value = prompt.invoke({"name": "World"})
        system, messages = custom_convert_to_anthropic_messages(prompt_value)
        assert system == "You are helpful"
        assert messages == [{"role": "user", "content": "Hello World"}]

    def test_multiple_system_messages(self):
        """Multiple system messages become a list of content blocks."""
        system, messages = custom_convert_to_anthropic_messages(
            [
                SystemMessage("You are helpful."),
                SystemMessage("Be concise."),
                HumanMessage("Hello"),
            ]
        )
        assert system == [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]
        assert messages == [{"role": "user", "content": "Hello"}]
