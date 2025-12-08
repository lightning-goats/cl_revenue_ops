"""
Prometheus Metrics Exporter module for cl-revenue-ops

Phase 2: Observability - Expose metrics for monitoring algorithmic decisions.

This module provides a lightweight, thread-safe Prometheus metrics exporter
using only the Python standard library (no prometheus_client or flask).

Features:
- Thread-safe metric storage using threading.Lock
- Support for Gauges (set value) and Counters (increment value)
- Support for labels (e.g., {channel_id="...", peer_id="..."})
- Background HTTP server for /metrics endpoint
- Standard Prometheus text format output

All metric names are prefixed with 'cl_revenue_' to avoid collisions.
"""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional, Any, Tuple
from collections import defaultdict


class MetricType:
    """Metric type constants."""
    GAUGE = "gauge"
    COUNTER = "counter"


class PrometheusExporter:
    """
    Lightweight Prometheus metrics exporter.
    
    Thread-safe implementation using standard library only.
    Supports gauges (set value) and counters (increment value) with labels.
    
    Usage:
        exporter = PrometheusExporter(port=9800)
        exporter.start_server()
        
        # Set a gauge
        exporter.set_gauge(
            "cl_revenue_channel_fee_ppm",
            500,
            {"channel_id": "123x1x1", "peer_id": "abc..."},
            "Current fee rate in PPM"
        )
        
        # Increment a counter
        exporter.inc_counter(
            "cl_revenue_rebalance_cost_total_sats",
            150,
            {"channel_id": "123x1x1"},
            "Total rebalancing costs"
        )
    """
    
    def __init__(self, port: int = 9800, plugin=None):
        """
        Initialize the Prometheus exporter.
        
        Args:
            port: HTTP server port (default: 9800)
            plugin: Optional plugin instance for logging
        """
        self.port = port
        self.plugin = plugin
        
        # Thread-safe storage
        self._lock = threading.Lock()
        
        # Storage structure:
        # {
        #   "metric_name": {
        #       "type": "gauge" | "counter",
        #       "help": "Description string",
        #       "values": {
        #           frozenset(labels.items()): value
        #       }
        #   }
        # }
        self._metrics: Dict[str, Dict[str, Any]] = {}
        
        # Server references
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
    
    def _log(self, message: str, level: str = 'info'):
        """Log a message using the plugin logger if available."""
        if self.plugin:
            self.plugin.log(message, level=level)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                  help_text: str = "") -> None:
        """
        Set a gauge metric value.
        
        Gauges can go up or down - they represent a current value.
        
        Args:
            name: Metric name (should start with 'cl_revenue_')
            value: The value to set
            labels: Optional dict of labels (e.g., {"channel_id": "123x1x1"})
            help_text: Description of the metric (only used on first set)
        """
        labels = labels or {}
        label_key = frozenset(labels.items())
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = {
                    "type": MetricType.GAUGE,
                    "help": help_text,
                    "values": {}
                }
            
            self._metrics[name]["values"][label_key] = value
    
    def inc_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None,
                    help_text: str = "") -> None:
        """
        Increment a counter metric.
        
        Counters only go up (or reset to zero). Use for cumulative values.
        
        Args:
            name: Metric name (should start with 'cl_revenue_')
            value: The amount to increment by (default: 1)
            labels: Optional dict of labels
            help_text: Description of the metric (only used on first set)
        """
        labels = labels or {}
        label_key = frozenset(labels.items())
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = {
                    "type": MetricType.COUNTER,
                    "help": help_text,
                    "values": {}
                }
            
            current = self._metrics[name]["values"].get(label_key, 0)
            self._metrics[name]["values"][label_key] = current + value
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Get the current value of a metric.
        
        Args:
            name: Metric name
            labels: Labels to match (must be exact match)
            
        Returns:
            Current metric value or None if not found
        """
        labels = labels or {}
        label_key = frozenset(labels.items())
        
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]["values"].get(label_key)
        return None
    
    def remove_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> bool:
        """
        Remove a metric or specific label combination.
        
        Useful for cleaning up metrics for closed channels.
        
        Args:
            name: Metric name
            labels: If provided, only remove this label combination.
                    If None, remove all values for this metric.
                    
        Returns:
            True if something was removed, False otherwise
        """
        labels = labels or {}
        
        with self._lock:
            if name not in self._metrics:
                return False
            
            if labels:
                label_key = frozenset(labels.items())
                if label_key in self._metrics[name]["values"]:
                    del self._metrics[name]["values"][label_key]
                    return True
                return False
            else:
                # Remove entire metric
                del self._metrics[name]
                return True
    
    def format_prometheus(self) -> str:
        """
        Format all metrics in Prometheus text format.
        
        Returns:
            String in Prometheus text exposition format
        """
        lines = []
        
        with self._lock:
            for name, metric in sorted(self._metrics.items()):
                metric_type = metric["type"]
                help_text = metric.get("help", "")
                values = metric["values"]
                
                # Add HELP line if we have help text
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                
                # Add TYPE line
                lines.append(f"# TYPE {name} {metric_type}")
                
                # Add metric values
                for label_key, value in sorted(values.items(), key=lambda x: str(x[0])):
                    if label_key:
                        # Format labels: {key1="value1", key2="value2"}
                        label_strs = [f'{k}="{v}"' for k, v in sorted(label_key)]
                        label_part = "{" + ", ".join(label_strs) + "}"
                        lines.append(f"{name}{label_part} {value}")
                    else:
                        lines.append(f"{name} {value}")
                
                # Blank line between metrics
                lines.append("")
        
        return "\n".join(lines)
    
    def _create_request_handler(self):
        """Create a request handler class with access to the exporter."""
        exporter = self
        
        class MetricsHandler(BaseHTTPRequestHandler):
            """HTTP request handler for /metrics endpoint."""
            
            def log_message(self, format, *args):
                """Override to use plugin logging instead of stderr."""
                # Suppress default logging to avoid noise
                pass
            
            def do_GET(self):
                """Handle GET requests."""
                try:
                    if self.path in ('/', '/metrics'):
                        try:
                            content = exporter.format_prometheus()
                            self.send_response(200)
                            self.send_header('Content-Type', 'text/plain; charset=utf-8')
                            self.send_header('Content-Length', str(len(content)))
                            self.end_headers()
                            self.wfile.write(content.encode('utf-8'))
                        except (BrokenPipeError, ConnectionResetError):
                            # Client disconnected before we finished - this is normal
                            pass
                        except Exception as e:
                            try:
                                self.send_response(500)
                                self.send_header('Content-Type', 'text/plain')
                                self.end_headers()
                                error_msg = f"Error generating metrics: {e}"
                                self.wfile.write(error_msg.encode('utf-8'))
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                    else:
                        self.send_response(404)
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b'Not Found. Try /metrics')
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected - silently ignore
                    pass
            
            def do_HEAD(self):
                """Handle HEAD requests."""
                try:
                    if self.path in ('/', '/metrics'):
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; charset=utf-8')
                        self.end_headers()
                    else:
                        self.send_response(404)
                        self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected - silently ignore
                    pass
        
        return MetricsHandler
    
    def start_server(self) -> bool:
        """
        Start the HTTP server in a background thread.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if self._running:
            self._log("Prometheus server already running")
            return True
        
        try:
            handler = self._create_request_handler()
            self._server = HTTPServer(('0.0.0.0', self.port), handler)
            
            # Set socket options to allow quick rebind
            self._server.socket.setsockopt(
                __import__('socket').SOL_SOCKET,
                __import__('socket').SO_REUSEADDR,
                1
            )
            
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="prometheus-exporter"
            )
            self._server_thread.start()
            self._running = True
            
            self._log(f"Prometheus metrics server started on port {self.port}")
            return True
            
        except OSError as e:
            if e.errno == 98:  # Address already in use
                self._log(
                    f"Prometheus port {self.port} is already in use. "
                    "Metrics will not be exported. Plugin continues without metrics.",
                    level='error'
                )
            else:
                self._log(f"Failed to start Prometheus server: {e}", level='error')
            return False
        except Exception as e:
            self._log(f"Unexpected error starting Prometheus server: {e}", level='error')
            return False
    
    def _run_server(self):
        """Run the HTTP server (called in background thread)."""
        try:
            self._server.serve_forever()
        except Exception as e:
            self._log(f"Prometheus server error: {e}", level='error')
            self._running = False
    
    def stop_server(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._running = False
            self._log("Prometheus metrics server stopped")
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running


# Metric name constants for consistency across modules
class MetricNames:
    """
    Standard metric names for cl-revenue-ops.
    
    All names are prefixed with 'cl_revenue_' to avoid collisions.
    """
    
    # Fee Controller metrics (Gauges)
    CHANNEL_FEE_PPM = "cl_revenue_channel_fee_ppm"
    CHANNEL_REVENUE_RATE_SATS_HR = "cl_revenue_channel_revenue_rate_sats_hr"
    
    # Rebalancer metrics (Counters)
    REBALANCE_COST_TOTAL_SATS = "cl_revenue_rebalance_cost_total_sats"
    REBALANCE_VOLUME_TOTAL_SATS = "cl_revenue_rebalance_volume_total_sats"
    REBALANCE_FAILURES_TOTAL = "cl_revenue_rebalance_failures_total"
    
    # Profitability Analyzer metrics (Gauges)
    CHANNEL_MARGINAL_ROI_PERCENT = "cl_revenue_channel_marginal_roi_percent"
    CHANNEL_CAPACITY_SATS = "cl_revenue_channel_capacity_sats"
    
    # Peer Reputation metrics (Gauges)
    PEER_REPUTATION_SCORE = "cl_revenue_peer_reputation_score"
    PEER_SUCCESS_COUNT = "cl_revenue_peer_success_count"
    PEER_FAILURE_COUNT = "cl_revenue_peer_failure_count"
    
    # Deadband Hysteresis metrics (Gauges)
    CHANNEL_IS_SLEEPING = "cl_revenue_channel_is_sleeping"
    
    # System health metrics (Gauges)
    SYSTEM_LAST_RUN_TIMESTAMP = "cl_revenue_system_last_run_timestamp_seconds"


# Help text for each metric
METRIC_HELP = {
    MetricNames.CHANNEL_FEE_PPM: "Current fee rate in parts per million",
    MetricNames.CHANNEL_REVENUE_RATE_SATS_HR: "Calculated revenue rate in sats per hour",
    MetricNames.REBALANCE_COST_TOTAL_SATS: "Total rebalancing fees paid in sats",
    MetricNames.REBALANCE_VOLUME_TOTAL_SATS: "Total volume moved via rebalancing in sats",
    MetricNames.REBALANCE_FAILURES_TOTAL: "Total number of failed rebalance attempts",
    MetricNames.CHANNEL_MARGINAL_ROI_PERCENT: "Marginal ROI percentage (operational profitability)",
    MetricNames.CHANNEL_CAPACITY_SATS: "Channel capacity in sats",
    MetricNames.PEER_REPUTATION_SCORE: "Peer reputation score (success rate 0.0 to 1.0)",
    MetricNames.PEER_SUCCESS_COUNT: "Total successful forwards from peer",
    MetricNames.PEER_FAILURE_COUNT: "Total failed forwards from peer",
    MetricNames.CHANNEL_IS_SLEEPING: "1 if channel is in Deadband Hysteresis (sleep mode), 0 otherwise",
    MetricNames.SYSTEM_LAST_RUN_TIMESTAMP: "Unix timestamp of last successful task run (for health monitoring)",
}
