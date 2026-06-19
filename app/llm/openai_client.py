from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core.config import settings


CONNECTION_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ProxyError",
}
NON_RECOVERABLE_ERROR_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
}
NON_RECOVERABLE_STATUS_CODES = {400, 401, 403, 404, 429}


class OpenAINonRecoverableError(RuntimeError):
    pass


class OpenAIProxyFallbackClient:
    """Prefer direct OpenAI calls, then retry once through the configured proxy."""

    def __init__(self, direct_client: Any, proxy_client: Any | None = None) -> None:
        self._direct_client = direct_client
        self._proxy_client = proxy_client

    def __getattr__(self, name: str) -> Any:
        direct_attr = getattr(self._direct_client, name)
        proxy_attr = getattr(self._proxy_client, name) if self._proxy_client is not None else None
        return _FallbackResource(direct_attr, proxy_attr)


class _FallbackResource:
    def __init__(self, direct_resource: Any, proxy_resource: Any | None = None) -> None:
        self._direct_resource = direct_resource
        self._proxy_resource = proxy_resource

    def __getattr__(self, name: str) -> Any:
        direct_attr = getattr(self._direct_resource, name)
        proxy_attr = getattr(self._proxy_resource, name) if self._proxy_resource is not None else None
        if callable(direct_attr):
            return _fallback_call(direct_attr, proxy_attr)
        if proxy_attr is None:
            return direct_attr
        return _FallbackResource(direct_attr, proxy_attr)


def _fallback_call(direct_call: Callable[..., Any], proxy_call: Any | None) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return direct_call(*args, **kwargs)
        except Exception as exc:
            if proxy_call is None or not is_openai_connection_error(exc):
                raise
            return proxy_call(*args, **kwargs)

    return call


def is_openai_connection_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if current.__class__.__name__ in CONNECTION_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def is_openai_non_recoverable_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if current.__class__.__name__ in NON_RECOVERABLE_ERROR_NAMES:
            return True
        status_code = getattr(current, "status_code", None)
        if status_code in NON_RECOVERABLE_STATUS_CODES:
            return True
        current = current.__cause__ or current.__context__
    return False


def summarize_openai_error(exc: Exception) -> str:
    status_code = openai_status_code(exc)
    message = str(exc)
    if status_code == 401 or "invalid_api_key" in message or "Incorrect API key" in message:
        return "OpenAI API Key 无效或已被撤销；请更新 .env 中的 OPENAI_API_KEY 后重启服务。"
    if status_code == 403:
        return "OpenAI API Key 没有当前模型或接口权限；请检查项目权限、模型权限和账单状态。"
    if status_code == 404:
        return "OpenAI 模型或接口不存在；请检查 OPENAI_MODEL / OPENAI_FAST_MODEL 配置。"
    if status_code == 429:
        return "OpenAI 额度不足或触发限速；请检查余额、项目限额，稍后重试或降低源数量。"
    if status_code == 400:
        return "OpenAI 请求被拒绝；通常是模型名、结构化输出 schema 或请求参数不兼容。"
    return f"OpenAI 调用失败且不可自动恢复：{exc.__class__.__name__}"


def openai_status_code(exc: Exception) -> int | None:
    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        current = current.__cause__ or current.__context__
    return None


def build_openai_client():
    try:
        from openai import DefaultHttpxClient, OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI client is not installed: {exc}") from exc

    direct_client = OpenAI(api_key=settings.openai_api_key)
    if not settings.openai_proxy:
        return direct_client
    proxy_client = OpenAI(
        api_key=settings.openai_api_key,
        http_client=DefaultHttpxClient(proxy=settings.openai_proxy),
    )
    return OpenAIProxyFallbackClient(direct_client, proxy_client)
