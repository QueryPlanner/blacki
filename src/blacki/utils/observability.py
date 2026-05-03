"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration using standard
OpenTelemetry environment variables for vendor-neutral operation.
"""

import logging
import os
import sys
import uuid

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
    Resource,
)


def configure_otel_resource(agent_name: str) -> None:
    """Configure OpenTelemetry resource via environment variables.

    Sets standard OTel resource attributes. Exporters are configured via
    standard OTel environment variables (OTEL_EXPORTER_OTLP_*) set by the user.

    Args:
        agent_name: Unique service identifier
    """
    print("🔭 Setting OpenTelemetry Resource attributes environment variable...")
    instance_id = f"worker-{os.getpid()}-{uuid.uuid4().hex}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{SERVICE_INSTANCE_ID}={instance_id},"
        f"{SERVICE_NAME}={agent_name},"
        f"{SERVICE_NAMESPACE}={os.getenv('TELEMETRY_NAMESPACE', 'local')},"
        f"{SERVICE_VERSION}={os.getenv('K_REVISION', 'local')}"
    )


def setup_logging(log_level: str) -> None:
    """Set up basic logging with optional OTLP log export.

    Configures Python logging to output to stdout. If OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
    is set, also exports logs via OTLP for correlation with traces.

    Args:
        log_level: Logging verbosity level as string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger with stdout handler
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set levels for some noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Set up OTLP log export if endpoint is configured
    logs_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if logs_endpoint:
        print(f"📊 Configuring OTLP log export to: {logs_endpoint}")

        # Create resource from OTEL_RESOURCE_ATTRIBUTES
        resource_attrs = {}
        if "OTEL_RESOURCE_ATTRIBUTES" in os.environ:
            for pair in os.environ["OTEL_RESOURCE_ATTRIBUTES"].split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    resource_attrs[key] = value

        resource = Resource.create(resource_attrs)

        # Set up logger provider with OTLP exporter
        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        set_logger_provider(provider)

        # Add OTLP handler to root logger
        handler = LoggingHandler(level=level)
        logging.getLogger().addHandler(handler)

        print("✅ OTLP log export configured")
