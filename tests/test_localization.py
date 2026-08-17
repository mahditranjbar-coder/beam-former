import numpy as np

from beam_former.config import AnalysisConfig, SignalConfig
from beam_former.geometry import angular_error_deg, circular_array
from beam_former.localization import localize_near_field
from beam_former.models import BeaconSource
from beam_former.signal import simulate_array_samples
from beam_former.spectral import covariance_at_bin, stft_snapshots


def test_near_field_localization_recovers_direction_and_range() -> None:
    signal = SignalConfig(duration_s=3.0, noise_std=0.004)
    analysis = AnalysisConfig()
    receivers = circular_array()
    position = np.array([1.25, -0.8, 1.45])
    source = BeaconSource(
        position=position,
        carrier_hz=1_400.0,
        payload=np.random.default_rng(1).integers(
            0, 2, signal.payload_length, dtype=np.int8
        ),
        amplitude=1.0,
        phase_rad=0.4,
        symbol_offset_s=0.002,
    )
    samples = simulate_array_samples(
        receivers, (source,), signal, rng=np.random.default_rng(9)
    )
    snapshots = stft_snapshots(samples, analysis.fft_size, analysis.fft_hop)
    bin_index = int(round(source.carrier_hz * analysis.fft_size / signal.sample_rate))
    covariance = covariance_at_bin(snapshots, bin_index, analysis)

    estimate = localize_near_field(
        source.carrier_hz, covariance, receivers, signal, analysis
    )

    assert angular_error_deg(estimate.position, position) < 2.0
    assert abs(estimate.range_m - np.linalg.norm(position)) < 0.3

