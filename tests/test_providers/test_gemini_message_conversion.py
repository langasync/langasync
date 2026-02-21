"""Test _convert_to_gemini_messages behavior on different input types.

This tests our conversion function that transforms LangChain messages into
Gemini REST API format (system_instruction + contents).
If these tests fail after a langchain upgrade, the underlying message types
have changed and our conversion needs updating.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langasync.providers.gemini import _convert_to_gemini_messages


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
        """HumanMessage with image URL is passed through as Gemini parts."""
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
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "url": "https://example.com/cat.jpg"},
                ],
            },
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
        """AI message with both text content and tool_calls includes both."""
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
                    {"text": "I'll check the weather."},
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

    def test_unsupported_message_type_raises(self):
        """Unsupported message type raises ValueError."""
        from langchain_core.messages import ChatMessage

        with pytest.raises(ValueError, match="Unsupported message type"):
            _convert_to_gemini_messages([ChatMessage(content="Hello", role="custom")])
