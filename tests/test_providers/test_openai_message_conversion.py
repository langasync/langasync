"""Test convert_to_openai_messages behavior on different input types."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
