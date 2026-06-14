"""OTel tracer/provider setup.

Langfuse (or any OTLP collector) is configured purely as an OTLP export
destination via `OtelConfig` -- nothing here is Langfuse-specific, and
`agent/core` never imports this module directly (only the composition root
and `OtelSink` do).
"""

from __future__ import annotations

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent.config import OtelConfig


def build_tracer_provider(config: OtelConfig) -> TracerProvider:
    """Build a `TracerProvider`. If `config.endpoint` is set, exports spans
    via OTLP/HTTP; otherwise spans are created but never exported."""
    provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
    if config.enabled and config.endpoint:
        exporter = OTLPSpanExporter(
            endpoint=_traces_endpoint(config.endpoint), headers=config.headers
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _traces_endpoint(base_endpoint: str) -> str:
    """OTLPSpanExporter does not append `/v1/traces` to an explicit
    `endpoint=`, unlike the generic `OTEL_EXPORTER_OTLP_ENDPOINT` env var."""
    return base_endpoint.rstrip("/") + "/v1/traces"
