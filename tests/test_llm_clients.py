from dms.llm.client import OpenAIChatClient, _extract_openai_chat_text


def test_extract_openai_chat_text_from_message_content() -> None:
    raw = {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"total_tokens": 3}}

    assert _extract_openai_chat_text(raw) == '{"ok": true}'


def test_extract_openai_chat_text_falls_back_to_text_choice() -> None:
    raw = {"choices": [{"text": "plain"}]}

    assert _extract_openai_chat_text(raw) == "plain"


def test_openai_chat_client_defaults_disable_thinking() -> None:
    client = OpenAIChatClient(api_key="token")

    assert client.enable_thinking is False


def test_openai_chat_client_preserves_v1_base_url() -> None:
    client = OpenAIChatClient(api_key="token", base_url="https://example.test/v1")
    endpoint = client.base_url
    if endpoint.endswith("/v1"):
        endpoint = endpoint + "/chat/completions"
    elif not endpoint.endswith("/v1/chat/completions"):
        endpoint = endpoint + "/v1/chat/completions"

    assert endpoint == "https://example.test/v1/chat/completions"
