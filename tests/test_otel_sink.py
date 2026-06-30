"""Phase 1 checkpoint: OtelSink projects a transcript onto a well-formed,
correctly-nested OTel span tree, and back-fills trace/span ids onto events."""

import sys
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.config import MCPServerConfig, OtelConfig
from agent.core.entrypoint import run_agent
from agent.core.events import Error, RunFinished
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.observability.otel import _validate_endpoint  # pyright: ignore[reportPrivateUsage]
from agent.observability.sink import OtelSink
from agent.skills.registry import EmptySkillRegistry

CASSETTE = Path(__file__).parent / "cassettes" / "echo_clock.json"

ECHO_CLOCK_SERVER = MCPServerConfig(
    name="echo-clock",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.echo_clock.server"],
)


async def test_otel_sink_produces_nested_span_tree() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    sink = OtelSink(tracer)

    model = ReplayModel(CASSETTE)
    permissions = AllowlistPolicy([AllowRule(server="echo-clock", tool="echo")])
    task = Task(
        id="echo-hello",
        system_prompt="You are a helpful agent with access to an echo tool.",
        messages=[Message(role="user", content=[TextBlock(text="Please echo 'hello'.")])],
    )

    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        result = await run_agent(
            task,
            model=model,
            tools=tools,
            skills=EmptySkillRegistry(),
            permissions=permissions,
            sink=sink,
        )

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}

    assert {
        "agent_run echo-hello",
        "step 0",
        "step 1",
        "chat replay",
        "execute_tool echo",
    } <= set(by_name)

    run_span = by_name["agent_run echo-hello"]
    step0 = by_name["step 0"]
    step1 = by_name["step 1"]
    chat_spans = [s for s in spans if s.name == "chat replay"]
    tool_span = by_name["execute_tool echo"]

    assert run_span.context is not None
    assert step0.context is not None
    assert step1.context is not None
    assert tool_span.context is not None

    assert run_span.parent is None
    assert step0.parent is not None and step0.parent.span_id == run_span.context.span_id
    assert step1.parent is not None and step1.parent.span_id == run_span.context.span_id

    step_span_ids = {step0.context.span_id, step1.context.span_id}
    assert len(chat_spans) == 2
    for chat in chat_spans:
        assert chat.parent is not None
        assert chat.parent.span_id in step_span_ids

    assert tool_span.parent is not None
    assert tool_span.parent.span_id == step0.context.span_id

    # The run-finished event was back-filled with the run span's ids.
    run_finished = [e for e in result.transcript if isinstance(e, RunFinished)]
    assert len(run_finished) == 1
    assert run_finished[0].trace_id == format(run_span.context.trace_id, "032x")
    assert run_finished[0].span_id == format(run_span.context.span_id, "016x")


# --- PR 2b: OTLP endpoint validation ---


def test_validate_endpoint_raises_for_http_non_loopback_with_auth() -> None:
    with pytest.raises(ValueError, match="allow_insecure"):
        _validate_endpoint(
            "http://langfuse-web:3000/api/public/otel",
            has_auth=True,
            allow_insecure=False,
        )


def test_validate_endpoint_passes_when_allow_insecure_true() -> None:
    _validate_endpoint(
        "http://langfuse-web:3000/api/public/otel",
        has_auth=True,
        allow_insecure=True,
    )


def test_validate_endpoint_passes_for_loopback_http_with_auth() -> None:
    _validate_endpoint("http://localhost:4317/", has_auth=True, allow_insecure=False)


def test_validate_endpoint_passes_for_https_with_auth() -> None:
    _validate_endpoint("https://otel.example.com/v1/traces", has_auth=True, allow_insecure=False)


def test_validate_endpoint_passes_without_auth_over_http() -> None:
    _validate_endpoint(
        "http://langfuse-web:3000/api/public/otel", has_auth=False, allow_insecure=False
    )


def test_otel_config_disabled_by_default() -> None:
    cfg = OtelConfig()
    assert cfg.enabled is False


# --- PR 2d: error message truncation ---


async def test_record_error_truncates_long_message() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    sink = OtelSink(tracer)

    long_message = "x" * 500
    error = Error(run_id="r1", step_index=None, where="test", message=long_message)

    run_span = tracer.start_span("run", attributes={})
    sink._run_spans["r1"] = run_span  # pyright: ignore[reportPrivateUsage]

    await sink.emit(error)
    run_span.end()

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    error_events = [e for e in finished[0].events if e.name == "error"]
    assert len(error_events) == 1
    assert len(error_events[0].attributes["message"]) == 200  # type: ignore[arg-type]


async def test_record_error_keeps_short_message_intact() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    sink = OtelSink(tracer)

    short_message = "something went wrong"
    error = Error(run_id="r2", step_index=None, where="test", message=short_message)

    run_span = tracer.start_span("run", attributes={})
    sink._run_spans["r2"] = run_span  # pyright: ignore[reportPrivateUsage]

    await sink.emit(error)
    run_span.end()

    finished = exporter.get_finished_spans()
    error_events = [e for e in finished[0].events if e.name == "error"]
    assert error_events[0].attributes["message"] == short_message  # type: ignore[index]
