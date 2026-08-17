import numpy as np

from beam_former.config import SignalConfig
from beam_former.geometry import circular_array
from beam_former.signal import generate_sources, simulate_array_samples


def test_generated_sources_are_distinct_and_samples_are_finite() -> None:
    config = SignalConfig(duration_s=0.1)
    source_rng = np.random.default_rng(4)
    sources = generate_sources(source_rng, 3, config)
    frequencies = np.array([source.carrier_hz for source in sources])
    separations = np.abs(frequencies[:, None] - frequencies[None, :])
    separations += np.eye(3) * 1e9

    samples = simulate_array_samples(
        circular_array(8, 0.12),
        sources,
        config,
        rng=np.random.default_rng(5),
        sample_count=800,
    )

    assert np.min(separations) >= config.carrier_separation_hz
    assert samples.shape == (8, 800)
    assert np.isfinite(samples).all()

