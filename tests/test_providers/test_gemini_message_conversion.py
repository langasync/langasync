"""Test _convert_to_gemini_messages behavior on different input types.

This tests our conversion function that transforms LangChain messages into
Gemini REST API format (system_instruction + contents).
If these tests fail after a langchain upgrade, the underlying message types
have changed and our conversion needs updating.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langasync.providers.gemini import _convert_content_part, _convert_to_gemini_messages


class TestConvertToGeminiMessages:
    """Test _convert_to_gemini_messages with different LanguageModelInput types."""

    def test_string_returns_user_message(self):
        system, contents = _convert_to_gemini_messages("Hello world")
        assert system is None
        assert contents == [{"role": "user", "parts": [{"text": "Hello world"}]}]

    def test_human_message_returns_user_role(self):
        system, contents = _convert_to_gemini_messages([HumanMessage("Hello")])
        assert system is None
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_single_human_message_without_list(self):
        system, contents = _convert_to_gemini_messages(HumanMessage("Hello"))
        assert system is None
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_single_system_message_without_list(self):
        system, contents = _convert_to_gemini_messages(SystemMessage("You are helpful"))
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == []

    def test_single_ai_message_without_list(self):
        system, contents = _convert_to_gemini_messages(AIMessage("Hi there"))
        assert system is None
        assert contents == [{"role": "model", "parts": [{"text": "Hi there"}]}]

    def test_system_message_extracted_separately(self):
        system, contents = _convert_to_gemini_messages(
            [SystemMessage("You are helpful"), HumanMessage("Hello")]
        )
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_ai_message_returns_model_role(self):
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("Hello"), AIMessage("Hi there")]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
        ]

    def test_list_of_messages_returns_list(self):
        system, contents = _convert_to_gemini_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage("Hello"),
                AIMessage("Hi!"),
            ]
        )
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]},
        ]

    def test_conversation_with_system(self):
        system, contents = _convert_to_gemini_messages(
            [
                SystemMessage("Be concise"),
                HumanMessage("What is 2+2?"),
                AIMessage("4"),
                HumanMessage("And 3+3?"),
            ]
        )
        assert system == {"parts": [{"text": "Be concise"}]}
        assert contents == [
            {"role": "user", "parts": [{"text": "What is 2+2?"}]},
            {"role": "model", "parts": [{"text": "4"}]},
            {"role": "user", "parts": [{"text": "And 3+3?"}]},
        ]

    def test_system_message_only(self):
        """System message alone (edge case)."""
        system, contents = _convert_to_gemini_messages([SystemMessage("You are helpful")])
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == []

    def test_multiple_system_messages_merged(self):
        """Multiple system messages get merged into one system_instruction."""
        system, contents = _convert_to_gemini_messages(
            [
                SystemMessage("You are helpful."),
                SystemMessage("Be concise."),
                HumanMessage("Hello"),
            ]
        )
        assert system == {"parts": [{"text": "You are helpful."}, {"text": "Be concise."}]}
        assert contents == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_multiple_human_messages(self):
        """Multiple consecutive human messages."""
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("Hello"), HumanMessage("How are you?")]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "user", "parts": [{"text": "How are you?"}]},
        ]

    def test_ai_message_only(self):
        """Single AI message."""
        system, contents = _convert_to_gemini_messages([AIMessage("Hi there")])
        assert system is None
        assert contents == [{"role": "model", "parts": [{"text": "Hi there"}]}]

    def test_empty_list(self):
        """Empty message list."""
        system, contents = _convert_to_gemini_messages([])
        assert system is None
        assert contents == []

    def test_returns_tuple(self):
        """Verify return type is tuple of (system_instruction, contents)."""
        result = _convert_to_gemini_messages([HumanMessage("Hello")])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_image_url_content(self):
        """HumanMessage with image URL is converted to Gemini file_data format."""
        system, contents = _convert_to_gemini_messages(
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
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this image."},
                    {
                        "file_data": {
                            "mime_type": "image/jpeg",
                            "file_uri": "https://example.com/cat.jpg",
                        }
                    },
                ],
            },
        ]

    def test_image_url_png(self):
        """Image URL with .png extension gets correct MIME type."""
        system, contents = _convert_to_gemini_messages(
            [HumanMessage(content=[{"type": "image", "url": "https://example.com/photo.png"}])]
        )
        assert contents[0]["parts"] == [
            {"file_data": {"mime_type": "image/png", "file_uri": "https://example.com/photo.png"}},
        ]

    def test_image_data_uri(self):
        """Image with data: URI is converted to inline_data."""
        system, contents = _convert_to_gemini_messages(
            [HumanMessage(content=[{"type": "image", "url": "data:image/png;base64,iVBOR"}])]
        )
        assert contents[0]["parts"] == [
            {"inline_data": {"mime_type": "image/png", "data": "iVBOR"}},
        ]

    def test_base64_image_content(self):
        """HumanMessage with base64 image gets converted to inline_data."""
        system, contents = _convert_to_gemini_messages(
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
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "What is this?"},
                    {"inline_data": {"mime_type": "image/png", "data": "iVBOR..."}},
                ],
            },
        ]

    def test_openai_image_url_format(self):
        """OpenAI-style image_url content is converted to Gemini format."""
        system, contents = _convert_to_gemini_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    ]
                ),
            ]
        )
        assert contents[0]["parts"] == [
            {"text": "What's this?"},
            {"file_data": {"mime_type": "image/jpeg", "file_uri": "https://example.com/img.jpg"}},
        ]

    def test_file_content(self):
        """File content (e.g. PDF) is converted to inline_data."""
        system, contents = _convert_to_gemini_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Summarize this."},
                        {"type": "file", "base64": "JVBERi0x", "mime_type": "application/pdf"},
                    ]
                ),
            ]
        )
        assert contents[0]["parts"] == [
            {"text": "Summarize this."},
            {"inline_data": {"mime_type": "application/pdf", "data": "JVBERi0x"}},
        ]

    def test_ai_message_with_tool_calls(self):
        """AI message with tool_calls gets converted to functionCall parts."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("What is the weather?"), ai_with_tools]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "What is the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                ],
            },
        ]

    def test_ai_message_with_tool_calls_and_text(self):
        """AI message with tool_calls ignores text content (matches upstream behavior)."""
        ai_with_tools = AIMessage(
            content="I'll check the weather.",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("What is the weather?"), ai_with_tools]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "What is the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                ],
            },
        ]

    def test_tool_message_becomes_function_response(self):
        """ToolMessage gets converted to functionResponse in user role."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        tool_response = ToolMessage(
            content='{"temperature": 72}', tool_call_id="call_123", name="get_weather"
        )
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("What is the weather?"), ai_with_tools, tool_response]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "What is the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temperature": 72},
                        }
                    }
                ],
            },
        ]

    def test_tool_message_non_json_string(self):
        """ToolMessage with non-JSON string content gets wrapped in output dict."""
        tool_response = ToolMessage(
            content="72 degrees", tool_call_id="call_123", name="get_weather"
        )
        system, contents = _convert_to_gemini_messages([tool_response])
        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"output": "72 degrees"},
                        }
                    }
                ],
            },
        ]

    def test_tool_message_without_name(self):
        """ToolMessage without name uses empty string."""
        tool_response = ToolMessage(content='{"result": "ok"}', tool_call_id="call_123")
        system, contents = _convert_to_gemini_messages([tool_response])
        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "",
                            "response": {"result": "ok"},
                        }
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
        system, contents = _convert_to_gemini_messages(
            [HumanMessage("Weather and time?"), ai_with_tools]
        )
        assert system is None
        assert contents == [
            {"role": "user", "parts": [{"text": "Weather and time?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                    {"functionCall": {"name": "get_time", "args": {"timezone": "EST"}}},
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
        system, contents = _convert_to_gemini_messages(prompt_value)
        assert system == {"parts": [{"text": "You are helpful"}]}
        assert contents == [{"role": "user", "parts": [{"text": "Hello World"}]}]
