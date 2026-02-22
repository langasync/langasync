"""Test _convert_to_bedrock_messages behavior on different input types.

Bedrock uses the same Anthropic Messages format under the hood, so these tests
mirror test_anthropic_message_conversion.py. If these tests fail after a
langchain-anthropic upgrade, the underlying _format_messages function signature
or behavior has changed and our adapter needs updating.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langasync.providers.bedrock.core import _convert_to_bedrock_messages
from langasync.providers.bedrock.model_providers import AnthropicBedrockProvider

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
PROVIDER = AnthropicBedrockProvider(MODEL_ID)


class TestConvertToBedrockMessages:
    """Test _convert_to_bedrock_messages with different LanguageModelInput types."""

    def test_string_returns_user_message(self):
        result = _convert_to_bedrock_messages("Hello world", PROVIDER)
        assert result == {"messages": [{"role": "user", "content": "Hello world"}]}

    def test_human_message_returns_user_role(self):
        result = _convert_to_bedrock_messages([HumanMessage("Hello")], PROVIDER)
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_single_human_message_without_list(self):
        result = _convert_to_bedrock_messages(HumanMessage("Hello"), PROVIDER)
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_single_system_message_without_list(self):
        result = _convert_to_bedrock_messages(SystemMessage("You are helpful"), PROVIDER)
        assert result == {"system": "You are helpful", "messages": []}

    def test_single_ai_message_without_list(self):
        result = _convert_to_bedrock_messages(AIMessage("Hi there"), PROVIDER)
        assert result == {"messages": [{"role": "assistant", "content": "Hi there"}]}

    def test_system_message_extracted_separately(self):
        result = _convert_to_bedrock_messages(
            [SystemMessage("You are helpful"), HumanMessage("Hello")], PROVIDER
        )
        assert result == {
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hello"}],
        }

    def test_ai_message_returns_assistant_role(self):
        result = _convert_to_bedrock_messages(
            [HumanMessage("Hello"), AIMessage("Hi there")], PROVIDER
        )
        assert result == {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }

    def test_list_of_messages_returns_list(self):
        result = _convert_to_bedrock_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage("Hello"),
                AIMessage("Hi!"),
            ],
            PROVIDER,
        )
        assert result == {
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        }

    def test_conversation_with_system(self):
        result = _convert_to_bedrock_messages(
            [
                SystemMessage("Be concise"),
                HumanMessage("What is 2+2?"),
                AIMessage("4"),
                HumanMessage("And 3+3?"),
            ],
            PROVIDER,
        )
        assert result == {
            "system": "Be concise",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }

    def test_system_message_only(self):
        """System message alone (edge case - needs at least one user message in practice)."""
        result = _convert_to_bedrock_messages([SystemMessage("You are helpful")], PROVIDER)
        assert result == {"system": "You are helpful", "messages": []}

    def test_multiple_human_messages(self):
        """Multiple consecutive human messages."""
        result = _convert_to_bedrock_messages(
            [HumanMessage("Hello"), HumanMessage("How are you?")], PROVIDER
        )
        # Anthropic merges consecutive same-role messages
        assert len(result["messages"]) >= 1
        assert result["messages"][0]["role"] == "user"

    def test_ai_message_only(self):
        """Single AI message."""
        result = _convert_to_bedrock_messages([AIMessage("Hi there")], PROVIDER)
        assert result == {"messages": [{"role": "assistant", "content": "Hi there"}]}

    def test_empty_list(self):
        """Empty message list."""
        result = _convert_to_bedrock_messages([], PROVIDER)
        assert result == {"messages": []}

    def test_returns_dict(self):
        """Verify return type is dict with messages key."""
        result = _convert_to_bedrock_messages([HumanMessage("Hello")], PROVIDER)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_image_url_content(self):
        """HumanMessage with image URL gets converted to Anthropic image source format."""
        result = _convert_to_bedrock_messages(
            [
                SystemMessage("You are helpful"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "url": "https://example.com/cat.jpg"},
                    ]
                ),
            ],
            PROVIDER,
        )
        assert result == {
            "system": "You are helpful",
            "messages": [
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
            ],
        }

    def test_base64_image_content(self):
        """HumanMessage with base64 image gets converted to Anthropic base64 source format."""
        result = _convert_to_bedrock_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "base64": "iVBOR...", "mime_type": "image/png"},
                    ]
                ),
            ],
            PROVIDER,
        )
        assert result == {
            "messages": [
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
            ],
        }

    def test_ai_message_with_tool_calls(self):
        """AI message with tool_calls gets converted to tool_use blocks."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        result = _convert_to_bedrock_messages(
            [HumanMessage("What is the weather?"), ai_with_tools], PROVIDER
        )
        assert result == {
            "messages": [
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
        }

    def test_tool_message_becomes_tool_result(self):
        """ToolMessage gets converted to tool_result block."""
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "NYC"}, "id": "call_123"}],
        )
        tool_response = ToolMessage(content="72 degrees", tool_call_id="call_123")
        result = _convert_to_bedrock_messages(
            [HumanMessage("What is the weather?"), ai_with_tools, tool_response],
            PROVIDER,
        )
        assert result == {
            "messages": [
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
        }
