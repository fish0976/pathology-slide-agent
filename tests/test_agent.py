import asyncio
from unittest.mock import patch

import httpx
import pytest

from pathology.agent import answer_question, call_tool
from pathology.config import Settings

REPORT = {
    "quality": {"tissue_fraction": 0.4, "focus_variance": 15, "blur_warning": True},
    "limitations": ["research only"],
    "regions": [],
    "summary": "演示结果",
    "mode": "demo",
    "sampling": {"grid_coverage": 0.2},
    "score_label": "演示分数",
}


def test_tool_allowlist():
    assert "error" in call_tool("execute_shell", REPORT)


def test_llm_tool_roundtrip(tmp_path):
    calls = []

    def handle(request):
        import json

        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            assert body["tool_choice"] == "required"
            assert body["thinking"] == {"type": "disabled"}
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "provider-required opaque field",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_quality", "arguments": "{}"}}
                ],
            }
        else:
            assert body["messages"][-1]["role"] == "tool"
            assert body["messages"][-2]["reasoning_content"] == "provider-required opaque field"
            assert body["tool_choice"] == "auto"
            message = {"content": "演示模式，组织占比 40%。"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    factory = httpx.AsyncClient
    with patch(
        "pathology.agent.httpx.AsyncClient",
        side_effect=lambda **kw: factory(transport=httpx.MockTransport(handle), **kw),
    ):
        answer = asyncio.run(answer_question("质控", REPORT, Settings(data_dir=tmp_path, llm_api_key="test")))
    assert answer["source"] == "llm"
    assert answer["tools"] == ["get_quality"]
    assert answer["provider"] == "DeepSeek"
    assert "reasoning_content" not in answer
    assert len(calls) == 2


def test_remote_failure_has_local_fallback(tmp_path):
    factory = httpx.AsyncClient
    with patch(
        "pathology.agent.httpx.AsyncClient",
        side_effect=lambda **kw: factory(transport=httpx.MockTransport(lambda _: httpx.Response(503)), **kw),
    ):
        answer = asyncio.run(answer_question("质控", REPORT, Settings(data_dir=tmp_path, llm_api_key="test")))
    assert answer["source"] == "local"
    assert answer["warning"]


@pytest.mark.parametrize("status,text", [(401, "认证失败"), (402, "余额不足"), (429, "过于频繁")])
def test_provider_errors_are_actionable_and_do_not_echo_body(tmp_path, status, text):
    factory = httpx.AsyncClient
    with patch(
        "pathology.agent.httpx.AsyncClient",
        side_effect=lambda **kw: factory(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, json={"error": "secret-provider-body"})
            ),
            **kw,
        ),
    ):
        answer = asyncio.run(answer_question("摘要", REPORT, Settings(data_dir=tmp_path, llm_api_key="test")))
    assert answer["source"] == "local"
    assert text in answer["warning"]
    assert "secret-provider-body" not in str(answer)


@pytest.mark.parametrize(
    "message",
    [
        None,
        {"content": "未查证的回答"},
        {"tool_calls": [{"id": "x", "function": {"name": "execute_shell", "arguments": "{}"}}]},
    ],
)
def test_malformed_or_unverified_response_falls_back(tmp_path, message):
    factory = httpx.AsyncClient
    with patch(
        "pathology.agent.httpx.AsyncClient",
        side_effect=lambda **kw: factory(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"choices": [{"message": message}]})
            ),
            **kw,
        ),
    ):
        answer = asyncio.run(answer_question("摘要", REPORT, Settings(data_dir=tmp_path, llm_api_key="test")))
    assert answer["source"] == "local"
    assert answer["warning"]
