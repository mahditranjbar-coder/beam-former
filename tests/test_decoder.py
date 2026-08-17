import numpy as np

from beam_former.config import AnalysisConfig, SignalConfig
from beam_former.decoder import decode_ask_packets
from beam_former.models import BeaconSource
from beam_former.signal import simulate_array_samples


def test_decoder_recovers_timing_and_payload_without_clock_oracle() -> None:
    signal = SignalConfig(duration_s=3.0, noise_std=0.003)
    payload = np.random.default_rng(2).integers(
        0, 2, signal.payload_length, dtype=np.int8
    )
    source = BeaconSource(
        position=np.array([1.4, 0.2, 1.0]),
        carrier_hz=1_237.0,
        payload=payload,
        amplitude=1.0,
        phase_rad=0.8,
        symbol_offset_s=0.0037,
    )
    samples = simulate_array_samples(
        np.zeros((1, 3)), (source,), signal, rng=np.random.default_rng(3)
    )[0]

    decoded = decode_ask_packets(samples, source.carrier_hz, signal, AnalysisConfig())

    assert decoded.locked
    assert decoded.packet_count >= 3
    assert np.array_equal(decoded.payload, payload)
    assert len(decoded.hard_bits) > signal.packet_length
    assert len(decoded.packet_starts) == decoded.packet_count
