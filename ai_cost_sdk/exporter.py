"""Non-blocking exporters."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Iterable

from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter


class BackgroundExporter:
    """Simple background event exporter with a bounded queue."""

    def __init__(
        self,
        *,
        json_path: str | None = None,
        maxsize: int = 10_000,
        batch_size: int = 200,
        flush_interval: float = 0.25,
    ) -> None:
        self.json_path = Path(json_path) if json_path else None
        if self.json_path:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=maxsize)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, event: dict) -> None:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            pass  # fail open

    def _run(self) -> None:
        buf: list[dict] = []
        last = time.time()
        while not self._stop.is_set():
            timeout = self.flush_interval - (time.time() - last)
            if timeout < 0:
                timeout = 0
            try:
                item = self.queue.get(timeout=timeout)
                buf.append(item)
                if len(buf) >= self.batch_size:
                    self._flush(buf)
                    buf.clear()
                    last = time.time()
            except queue.Empty:
                if buf:
                    self._flush(buf)
                    buf.clear()
                    last = time.time()
        if buf:
            self._flush(buf)

    def _flush(self, batch: Iterable[dict]) -> None:
        if self.json_path:
            with self.json_path.open("a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


class FileSpanExporter(SpanExporter):
    """Minimal JSON lines span exporter."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8") as f:
            for span in spans:
                data = {
                    "name": span.name,
                    "context": {
                        "trace_id": format(span.context.trace_id, "032x"),
                        "span_id": format(span.context.span_id, "016x"),
                    },
                    "attributes": span.attributes,
                }
                f.write(json.dumps(data) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass
