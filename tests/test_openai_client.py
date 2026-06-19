from app.core.config import effective_openai_api_key, settings, use_request_openai_api_key
from app.llm import document_filter as document_filter_module
from app.llm.document_filter import DocumentRelevanceFilter
from app.llm.openai_client import OpenAIProxyFallbackClient, build_openai_client
from app.llm.openai_client import (
    OpenAINonRecoverableError,
    is_openai_non_recoverable_error,
    summarize_openai_error,
)
from app.parsers.models import ParsedDocument


class APIConnectionError(Exception):
    pass


class AuthenticationError(Exception):
    status_code = 401


def test_openai_client_accepts_openai_only_proxy(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_proxy", "http://127.0.0.1:7890")

    client = build_openai_client()

    assert client is not None


def test_request_scoped_openai_key_overrides_settings(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")

    assert effective_openai_api_key() == ""
    with use_request_openai_api_key("request-key"):
        assert effective_openai_api_key() == "request-key"
    assert effective_openai_api_key() == ""


def test_fallback_client_prefers_direct_when_direct_succeeds():
    calls = []
    direct = FakeClient(response="direct-ok", calls=calls, label="direct")
    proxy = FakeClient(response="proxy-ok", calls=calls, label="proxy")
    client = OpenAIProxyFallbackClient(direct, proxy)

    response = client.responses.create(model="test")

    assert response == "direct-ok"
    assert calls == ["direct"]


def test_nonrecoverable_openai_errors_are_sanitized():
    exc = AuthenticationError("Incorrect API key provided: sk-proj-secret")

    assert is_openai_non_recoverable_error(exc)
    message = summarize_openai_error(exc)
    assert "OpenAI API Key 无效" in message
    assert "sk-proj-secret" not in message


def test_live_document_filter_fails_fast_on_invalid_openai_key(monkeypatch):
    monkeypatch.setattr(document_filter_module.settings, "openai_api_key", "test-key")

    class FailingResponses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AuthenticationError("Incorrect API key provided: sk-proj-secret")

    class FailingClient:
        responses = FailingResponses()

    monkeypatch.setattr(document_filter_module, "build_openai_client", lambda: FailingClient())
    document = ParsedDocument(
        source_id="fixture",
        url="fixture://doc",
        title="张江产业规划",
        text="张江产业规划和住宅用地相关公告。",
    )

    try:
        DocumentRelevanceFilter(live=True).classify(document)
    except OpenAINonRecoverableError as exc:
        assert "OpenAI API Key 无效" in str(exc)
        assert "sk-proj-secret" not in str(exc)
    else:
        raise AssertionError("Expected invalid OpenAI key to fail fast")


def test_live_document_filter_falls_back_on_transient_triage_error(monkeypatch):
    monkeypatch.setattr(document_filter_module.settings, "openai_api_key", "test-key")

    class TransientResponses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("temporary triage failure")

    class TransientClient:
        responses = TransientResponses()

    monkeypatch.setattr(document_filter_module, "build_openai_client", lambda: TransientClient())
    document = ParsedDocument(
        source_id="fixture",
        url="fixture://doc",
        title="临港住宅用地公告",
        text="临港住宅用地和产业项目公告。",
    )

    relevance = DocumentRelevanceFilter(live=True).classify(document)

    assert relevance.should_extract is True
    assert "住宅" in relevance.categories


def test_fallback_client_uses_proxy_after_direct_connection_error():
    calls = []
    direct = FakeClient(error=APIConnectionError("Connection error."), calls=calls, label="direct")
    proxy = FakeClient(response="proxy-ok", calls=calls, label="proxy")
    client = OpenAIProxyFallbackClient(direct, proxy)

    response = client.responses.create(model="test")

    assert response == "proxy-ok"
    assert calls == ["direct", "proxy"]


def test_fallback_client_does_not_retry_non_connection_error():
    calls = []
    direct = FakeClient(error=ValueError("invalid schema"), calls=calls, label="direct")
    proxy = FakeClient(response="proxy-ok", calls=calls, label="proxy")
    client = OpenAIProxyFallbackClient(direct, proxy)

    try:
        client.responses.create(model="test")
    except ValueError as exc:
        assert str(exc) == "invalid schema"
    else:
        raise AssertionError("Expected ValueError")

    assert calls == ["direct"]


class FakeClient:
    def __init__(
        self,
        *,
        calls: list[str],
        label: str,
        response: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = FakeResource(calls=calls, label=label, response=response, error=error)


class FakeResource:
    def __init__(
        self,
        *,
        calls: list[str],
        label: str,
        response: str | None,
        error: Exception | None,
    ) -> None:
        self.calls = calls
        self.label = label
        self.response = response
        self.error = error

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(self.label)
        if self.error:
            raise self.error
        return self.response
