"""OTel tracer/provider setup.

Langfuse (or any OTLP collector) is configured purely as an OTLP export
destination via `OtelConfig` -- nothing here is Langfuse-specific, and
`agent/core` never imports this module directly (only the composition root
and `OtelSink` do).
"""

from __future__ import annotations

from urllib.parse import urlparse

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent.config import OtelConfig

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def build_tracer_provider(config: OtelConfig) -> TracerProvider:
    """Build a `TracerProvider`. If `config.endpoint` is set, exports spans
    via OTLP/HTTP; otherwise spans are created but never exported."""
    provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
    if config.enabled and config.endpoint:
        _validate_endpoint(
            config.endpoint,
            has_auth=bool(config.headers),
            allow_insecure=config.allow_insecure,
        )
        exporter = OTLPSpanExporter(
            endpoint=_traces_endpoint(config.endpoint), headers=config.headers
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _validate_endpoint(endpoint: str, *, has_auth: bool, allow_insecure: bool) -> None:
    """Refuse to export auth headers over plaintext to non-loopback hosts."""
    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
    is_loopback = host in _LOOPBACK_HOSTS
    if has_auth and not is_loopback and scheme not in ("https", "grpcs") and not allow_insecure:
        raise ValueError(
            f"OTLP endpoint '{endpoint}' sends auth headers over plaintext HTTP to a "
            "non-loopback host. Use https:// or set otel.allow_insecure = true for "
            "internal container networks where TLS is terminated at the network layer."
        )


def _traces_endpoint(base_endpoint: str) -> str:
    """OTLPSpanExporter does not append `/v1/traces` to an explicit
    `endpoint=`, unlike the generic `OTEL_EXPORTER_OTLP_ENDPOINT` env var."""
    return base_endpoint.rstrip("/") + "/v1/traces"
