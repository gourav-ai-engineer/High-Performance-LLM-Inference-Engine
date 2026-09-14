from telemetry.prometheus_exporter import (
    inference_input_tokens,
    inference_output_tokens,
    inference_requests_total,
    metrics_payload,
)


def test_prometheus_payload_contains_core_metrics():
    before = inference_requests_total._value.get()
    inference_requests_total.inc()
    inference_input_tokens.observe(12)
    inference_output_tokens.observe(8)
    payload = metrics_payload().decode("utf-8")
    assert "inference_requests_total" in payload
    assert "inference_input_tokens" in payload
    assert "inference_output_tokens" in payload
    assert inference_requests_total._value.get() == before + 1
