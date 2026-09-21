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
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_quality", "arguments": "{}"}}
                ],
            }
        else:
            assert body["messages"][-1]["role"] == "tool"
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
