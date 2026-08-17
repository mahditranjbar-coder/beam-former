import numpy as np

from beam_former.config import AnalysisConfig, SignalConfig
from beam_former.geometry import circular_array
from beam_former.models import BeaconSource
from beam_former.signal import simulate_array_samples
from beam_former.spectral import (
    averaged_positive_spectrum,
    detect_carriers,
    stft_snapshots,
)


def test_detector_finds_only_the_transmitted_carriers() -> None:
    signal = SignalConfig(duration_s=2.0, noise_std=0.005)
    analysis = AnalysisConfig(max_candidates=4, minimum_snr_db=10.0)
    carriers = (900.0, 1_350.0, 1_900.0)
    sources = tuple(
        BeaconSource(
            position=np.array([1.0 + index * 0.2, 0.4, 1.2]),
            carrier_hz=carrier,
            payload=np.tile([0, 1], signal.payload_length // 2).astype(np.int8),
            amplitude=1.0,
            phase_rad=0.3 * index,
            symbol_offset_s=0.001 * index,
        )
        for index, carrier in enumerate(carriers)
    )
    samples = simulate_array_samples(
        circular_array(), sources, signal, rng=np.random.default_rng(11)
    )
    snapshots = stft_snapshots(samples, analysis.fft_size, analysis.fft_hop)
    frequencies, power = averaged_positive_spectrum(snapshots, signal.sample_rate)
    detections = detect_carriers(frequencies, power, signal, analysis)

    found = np.array(sorted(item.frequency_hz for item in detections))
    assert len(found) == len(carriers)
    assert np.max(np.abs(found - carriers)) < 3.0

