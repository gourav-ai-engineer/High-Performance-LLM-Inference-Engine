from api.openai_models import ChatCompletionRequest, ChatMessage


def test_chat_prompt_preserves_message_order():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="Hello"),
        ],
    )
    assert request.prompt() == "system: You are concise.\nuser: Hello"


def test_chat_request_defaults_are_safe():
    request = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="Hello")]
    )
    assert request.max_tokens == 32
    assert request.temperature == 0.0
    assert request.top_p == 1.0
    assert request.stream is False
