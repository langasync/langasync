"""Test convert_to_openai_messages behavior on different input types."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_openai_messages


class TestConvertToOpenAIMessages:
    """Test convert_to_openai_messages with different LanguageModelInput types."""

    def test_string_returns_dict(self):
        result = convert_to_openai_messages("Hello world")
        assert result == {"role": "user", "content": "Hello world"}

    def test_human_message_returns_dict(self):
        result = convert_to_openai_messages(HumanMessage("Hello"))
        assert result == {"role": "user", "content": "Hello"}

    def test_system_message_returns_dict(self):
        result = convert_to_openai_messages(SystemMessage("You are helpful"))
        assert result == {"role": "system", "content": "You are helpful"}

    def test_ai_message_returns_dict(self):
        result = convert_to_openai_messages(AIMessage("Hi there"))
        assert result == {"role": "assistant", "content": "Hi there"}

    def test_list_of_messages_returns_list(self):
        result = convert_to_openai_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage("Hello"),
                AIMessage("Hi!"),
            ]
        )
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_dict_passes_through(self):
        result = convert_to_openai_messages({"role": "user", "content": "Hello"})
        assert result == {"role": "user", "content": "Hello"}

    def test_list_of_dicts_passes_through(self):
        result = convert_to_openai_messages(
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert result == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_ai_message_with_tool_calls(self):
        """AI message with tool_calls gets converted to function calls."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        result = convert_to_openai_messages([HumanMessage("What is the weather?"), ai_with_tools])
        assert result == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
                "content": "",
            },
        ]

    def test_image_url_content(self):
        """HumanMessage with image URL gets converted to OpenAI image_url format."""
        result = convert_to_openai_messages(
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
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                ],
            },
        ]

    def test_base64_image_content(self):
        """HumanMessage with base64 image gets converted to OpenAI image_url format."""
        result = convert_to_openai_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "base64": "iVBOR...", "mime_type": "image/png"},
                    ]
                ),
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR..."},
                    },
                ],
            },
        ]

    def test_file_content(self):
        """HumanMessage with base64 file gets converted to OpenAI file format."""
        result = convert_to_openai_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Summarize."},
                        {
                            "type": "file",
                            "base64": "JVBERi0xLjQK",
                            "mime_type": "application/pdf",
                            "filename": "document.pdf",
                        },
                    ]
                ),
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize."},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                            "filename": "document.pdf",
                        },
                    },
                ],
            },
        ]

    def test_tool_message(self):
        """ToolMessage gets converted to tool role."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        tool_response = ToolMessage(content="72 degrees", tool_call_id="call_123")
        result = convert_to_openai_messages(
            [HumanMessage("What is the weather?"), ai_with_tools, tool_response]
        )
        assert result == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
                "content": "",
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "72 degrees"},
        ]

    def test_ai_message_with_text_and_tool_calls(self):
        """AI message with both text content and tool_calls includes both."""
        ai_with_tools = AIMessage(
            content="I'll check the weather.",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        result = convert_to_openai_messages([HumanMessage("What is the weather?"), ai_with_tools])
        assert result == [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "I'll check the weather.",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
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
        result = convert_to_openai_messages([HumanMessage("Weather and time?"), ai_with_tools])
        assert result == [
            {"role": "user", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "get_time", "arguments": '{"timezone": "EST"}'},
                    },
                ],
            },
        ]

    def test_data_uri_image(self):
        """Data URI image is preserved as image_url."""
        result = convert_to_openai_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "url": "data:image/png;base64,iVBOR"},
                    ]
                ),
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                ],
            },
        ]

    def test_openai_image_url_format(self):
        """OpenAI-style image_url content passes through."""
        result = convert_to_openai_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    ]
                ),
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
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
        result = convert_to_openai_messages(prompt_value)
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello World"},
        ]

    def test_multiple_system_messages(self):
        """Multiple system messages stay as separate entries."""
        result = convert_to_openai_messages(
            [
                SystemMessage("You are helpful."),
                SystemMessage("Be concise."),
                HumanMessage("Hello"),
            ]
        )
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
