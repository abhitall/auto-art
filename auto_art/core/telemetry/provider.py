"""
OpenTelemetry-based telemetry provider.

Production-grade observability with three pillars:
- **Traces**: Distributed tracing with span context propagation
- **Metrics**: Counters, histograms, gauges for runtime statistics
- **Logs**: Structured JSON logging with automatic trace correlation

Supports OTLP export to any OpenTelemetry-compatible backend
(Jaeger, Zipkin, Grafana Tempo, Datadog, etc.).

Usage:
    from auto_art.core.telemetry import TelemetryProvider

    tp = TelemetryProvider(service_name="auto-art", environment="production")
    tp.initialize()

    with tp.trace_span("evaluate_model", attributes={"model": "resnet50"}):
        # ... evaluation logic ...
        tp.record_metric("attacks_executed", 5)
        tp.log("Evaluation complete", level="info", extra={"score": 95.0})
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class ExportTarget(Enum):
    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"


@dataclass
class TelemetryConfig:
    """Configuration for the telemetry provider."""
    service_name: str = "auto-art"
    service_version: str = "0.2.0"
    environment: str = "development"
    export_target: ExportTarget = ExportTarget.CONSOLE
    otlp_endpoint: str = "http://localhost:4317"
    enable_traces: bool = True
    enable_metrics: bool = True
    enable_logs: bool = True
    log_level: str = "INFO"
    batch_export: bool = True
    resource_attributes: Dict[str, str] = field(default_factory=dict)


_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
    )
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor as _LogInst
        _USE_INSTRUMENTATION_HANDLER = True
    except ImportError:
        _USE_INSTRUMENTATION_HANDLER = False

    try:
        from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
        _ConsoleLogExp = ConsoleLogRecordExporter
    except ImportError:
        _ConsoleLogExp = ConsoleLogExporter

    from opentelemetry.sdk._logs import LoggingHandler
    _OTEL_AVAILABLE = True
except ImportError:
    pass

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter
    _OTLP_GRPC_AVAILABLE = True
except ImportError:
    _OTLP_GRPC_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpMetricExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter
    _OTLP_HTTP_AVAILABLE = True
except ImportError:
    _OTLP_HTTP_AVAILABLE = False


class TelemetryProvider:
    """Production-grade telemetry with traces, metrics, and structured logs.

    When OpenTelemetry SDK is installed (pip install auto-art[telemetry]),
    provides full distributed tracing with OTLP export. Falls back to
    Python stdlib logging when OTel is unavailable.
    """

    _instance: Optional[TelemetryProvider] = None

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self._tracer: Any = None
        self._meter: Any = None
        self._logger_provider: Any = None
        self._structured_logger: Optional[logging.Logger] = None
        self._initialized = False
        self._counters: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls, config: Optional[TelemetryConfig] = None) -> TelemetryProvider:
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def initialize(self) -> None:
        """Initialize the telemetry provider with configured exporters."""
        if self._initialized:
            return

        self._setup_structured_logger()

        if _OTEL_AVAILABLE:
            resource = self._build_resource()
            if self.config.enable_traces:
                self._setup_traces(resource)
            if self.config.enable_metrics:
                self._setup_metrics(resource)
            if self.config.enable_logs:
                self._setup_log_provider(resource)
            self._structured_logger.info(
                "OpenTelemetry initialized",
                extra={
                    "export_target": self.config.export_target.value,
                    "traces": self.config.enable_traces,
                    "metrics": self.config.enable_metrics,
                    "logs": self.config.enable_logs,
                },
            )
        else:
            self._structured_logger.info(
                "OpenTelemetry SDK not available — using stdlib logging fallback. "
                "Install with: pip install auto-art[telemetry]"
            )

        self._initialized = True

    def _build_resource(self) -> Any:
        attrs = {
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
            **self.config.resource_attributes,
        }
        return Resource.create(attrs)

    def _setup_structured_logger(self) -> None:
        self._structured_logger = logging.getLogger(f"auto_art.telemetry.{self.config.service_name}")
        self._structured_logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        if not self._structured_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","message":"%(message)s"'
                '%(otel_extra)s}',
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            formatter.default_msec_format = "%s.%03dZ"
            handler.setFormatter(_JsonFormatter())
            self._structured_logger.addHandler(handler)

    def _setup_traces(self, resource: Any) -> None:
        provider = TracerProvider(resource=resource)
        exporter = self._get_span_exporter()
        if self.config.batch_export:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer(self.config.service_name, self.config.service_version)

    def _setup_metrics(self, resource: Any) -> None:
        exporter = self._get_metric_exporter()
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)
        self._meter = otel_metrics.get_meter(self.config.service_name, self.config.service_version)

    def _setup_log_provider(self, resource: Any) -> None:
        exporter = self._get_log_exporter()
        self._logger_provider = LoggerProvider(resource=resource)
        self._logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

        otel_handler = LoggingHandler(
            level=logging.NOTSET,
            logger_provider=self._logger_provider,
        )
        logging.getLogger().addHandler(otel_handler)

    def _get_span_exporter(self) -> Any:
        if self.config.export_target == ExportTarget.OTLP_GRPC and _OTLP_GRPC_AVAILABLE:
            return GrpcSpanExporter(endpoint=self.config.otlp_endpoint)
        if self.config.export_target == ExportTarget.OTLP_HTTP and _OTLP_HTTP_AVAILABLE:
            return HttpSpanExporter(endpoint=self.config.otlp_endpoint)
        return ConsoleSpanExporter()

    def _get_metric_exporter(self) -> Any:
        if self.config.export_target == ExportTarget.OTLP_GRPC and _OTLP_GRPC_AVAILABLE:
            return GrpcMetricExporter(endpoint=self.config.otlp_endpoint)
        if self.config.export_target == ExportTarget.OTLP_HTTP and _OTLP_HTTP_AVAILABLE:
            return HttpMetricExporter(endpoint=self.config.otlp_endpoint)
        return ConsoleMetricExporter()

    def _get_log_exporter(self) -> Any:
        if self.config.export_target == ExportTarget.OTLP_GRPC and _OTLP_GRPC_AVAILABLE:
            return GrpcLogExporter(endpoint=self.config.otlp_endpoint)
        if self.config.export_target == ExportTarget.OTLP_HTTP and _OTLP_HTTP_AVAILABLE:
            return HttpLogExporter(endpoint=self.config.otlp_endpoint)
        return _ConsoleLogExp()

    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Create a traced span with automatic timing and error capture."""
        if self._tracer is not None and _OTEL_AVAILABLE:
            with self._tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                try:
                    yield span
                except Exception as e:
                    span.set_status(otel_trace.StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        else:
            start = time.time()
            try:
                yield None
            finally:
                duration = (time.time() - start) * 1000
                if self._structured_logger:
                    self._structured_logger.debug(
                        f"span:{name} duration={duration:.1f}ms",
                        extra={"span_name": name, "duration_ms": duration, **(attributes or {})},
                    )

    def record_metric(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value (counter increment)."""
        if self._meter is not None and _OTEL_AVAILABLE:
            if name not in self._counters:
                self._counters[name] = self._meter.create_counter(
                    name=f"auto_art.{name}",
                    description=f"Auto-ART metric: {name}",
                )
            self._counters[name].add(value, attributes=attributes or {})
        elif self._structured_logger:
            self._structured_logger.info(
                f"metric:{name}={value}",
                extra={"metric_name": name, "metric_value": value, **(attributes or {})},
            )

    def record_histogram(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation (durations, sizes, etc.)."""
        if self._meter is not None and _OTEL_AVAILABLE:
            if name not in self._histograms:
                self._histograms[name] = self._meter.create_histogram(
                    name=f"auto_art.{name}",
                    description=f"Auto-ART histogram: {name}",
                )
            self._histograms[name].record(value, attributes=attributes or {})
        elif self._structured_logger:
            self._structured_logger.info(
                f"histogram:{name}={value}",
                extra={"histogram_name": name, "histogram_value": value, **(attributes or {})},
            )

    def log(self, message: str, level: str = "info", extra: Optional[Dict[str, Any]] = None) -> None:
        """Emit a structured log with automatic trace correlation."""
        if self._structured_logger:
            log_fn = getattr(self._structured_logger, level.lower(), self._structured_logger.info)
            log_fn(message, extra=extra or {})

    def shutdown(self) -> None:
        """Flush and shut down all telemetry providers."""
        if _OTEL_AVAILABLE:
            provider = otel_trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
            mprov = otel_metrics.get_meter_provider()
            if hasattr(mprov, "shutdown"):
                mprov.shutdown()
            if self._logger_provider and hasattr(self._logger_provider, "shutdown"):
                self._logger_provider.shutdown()
        self._initialized = False


class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter with OpenTelemetry trace correlation."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "otelTraceID") and record.otelTraceID != "0":
            log_data["trace_id"] = record.otelTraceID
        if hasattr(record, "otelSpanID") and record.otelSpanID != "0":
            log_data["span_id"] = record.otelSpanID

        skip_keys = {
            "message", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "name",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "exc_info", "exc_text",
            "otelTraceID", "otelSpanID", "otelTraceSampled", "otelServiceName",
            "otel_extra", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, default=str)
