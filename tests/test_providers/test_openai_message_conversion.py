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
