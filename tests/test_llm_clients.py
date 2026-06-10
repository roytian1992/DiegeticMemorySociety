import json

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


def test_openai_chat_client_can_send_thinking_disabled_payload(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode("utf-8")

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = OpenAIChatClient(
        api_key="token",
        base_url="https://example.test/v1",
        model="gpt-5.4-mini",
        thinking={"type": "disabled"},
        include_chat_template_kwargs=False,
    ).complete("Say OK")

    assert result.text == "OK"
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in calls[0]


def test_openai_chat_client_stream_fallback_when_non_stream_content_is_empty(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.payload.encode("utf-8")

    calls = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        calls.append(body)
        if body.get("stream"):
            payload = "\n".join(
                [
                    'data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0}],"model":"gpt-5.4-mini"}',
                    'data: {"choices":[{"delta":{"content":"OK"},"index":0}],"model":"gpt-5.4-mini"}',
                    'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"model":"gpt-5.4-mini"}',
                    'data: {"choices":[],"usage":{"total_tokens":5}}',
                    "data: [DONE]",
                ]
            )
            return FakeResponse(payload)
        return FakeResponse(
            json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 3},
                }
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = OpenAIChatClient(
        api_key="token",
        base_url="https://example.test/v1",
        model="gpt-5.4-mini",
    ).complete("Say OK")

    assert result.text == "OK"
    assert result.raw_response["stream_fallback_used"] is True
    assert calls[0].get("stream") is None
    assert calls[1]["stream"] is True
