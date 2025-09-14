from ai_cost_sdk.exporter import BackgroundExporter


def test_overflow_drops():
    exp = BackgroundExporter(json_path=None, flush_interval=10)
    for _ in range(10_000):
        exp.submit({"a": 1})
    exp.submit({"a": 2})  # overflow
    assert exp.queue.qsize() <= 10_000
    exp.shutdown()
