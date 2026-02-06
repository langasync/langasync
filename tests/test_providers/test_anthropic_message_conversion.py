"""Test custom_convert_to_anthropic_messages behavior on different input types.

This tests our adapter function that wraps langchain-anthropic's _format_messages.
If these tests fail after a langchain-anthropic upgrade, the underlying function
signature or behavior has changed and our adapter needs updating.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
