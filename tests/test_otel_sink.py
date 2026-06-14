"""Phase 1 checkpoint: OtelSink projects a transcript onto a well-formed,
correctly-nested OTel span tree, and back-fills trace/span ids onto events."""

import sys
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.config import MCPServerConfig
from agent.core.entrypoint import run_agent
from agent.core.events import RunFinished
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
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
