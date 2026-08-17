"""Deterministic multi-beacon signal generation."""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_PREAMBLE, SignalConfig
from .geometry import unit_vector
from .models import BeaconSource, ComplexArray


def _sample_distinct_carriers(
    rng: np.random.Generator, count: int, config: SignalConfig
) -> np.ndarray:
    carriers: list[float] = []
    for _ in range(100_000):
        candidate = float(
            rng.uniform(config.carrier_min_hz, config.carrier_max_hz)
        )
        if all(
            abs(candidate - existing) >= config.carrier_separation_hz
            for existing in carriers
        ):
            carriers.append(candidate)
            if len(carriers) == count:
                return np.sort(np.asarray(carriers))
    raise ValueError("carrier band is too narrow for the requested source count")


def generate_sources(
    rng: np.random.Generator,
    count: int,
    config: SignalConfig,
    *,
    range_limits_m: tuple[float, float] = (1.2, 3.3),
) -> tuple[BeaconSource, ...]:
    if count < 1:
        raise ValueError("source count must be positive")
    carriers = _sample_distinct_carriers(rng, count, config)
    sources: list[BeaconSource] = []
    used_directions: list[np.ndarray] = []

    for carrier in carriers:
        for _ in range(10_000):
            azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
            elevation = float(np.deg2rad(rng.uniform(15.0, 70.0)))
            direction = unit_vector(azimuth, elevation)
            if all(
                np.degrees(np.arccos(np.clip(direction @ other, -1.0, 1.0)))
                >= 18.0
                for other in used_directions
            ):
                break
        else:
            raise RuntimeError("could not place spatially separated sources")
        used_directions.append(direction)
        distance = float(rng.uniform(*range_limits_m))
        payload = rng.integers(0, 2, config.payload_length, dtype=np.int8)
        sources.append(
            BeaconSource(
                position=(distance * direction).astype(np.float64),
                carrier_hz=float(carrier),
                payload=payload,
                amplitude=float(rng.uniform(0.9, 1.2)),
                phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
                symbol_offset_s=float(rng.uniform(0.0, 1.0 / config.symbol_rate)),
            )
        )
    return tuple(sources)


def packet_bits(source: BeaconSource) -> np.ndarray:
    return np.concatenate((DEFAULT_PREAMBLE, source.payload))


def simulate_array_samples(
    receiver_positions: np.ndarray,
    sources: tuple[BeaconSource, ...],
    config: SignalConfig,
    *,
    rng: np.random.Generator,
    start_sample: int = 0,
    sample_count: int | None = None,
) -> ComplexArray:
    """Generate complex analytic pressure at every receiver element.

    Carrier and ASK envelope both use retarded time. The small array aperture
    keeps differential envelope delay well below one symbol while preserving
    physically consistent carrier phase.
    """

    if sample_count is None:
        sample_count = int(round(config.duration_s * config.sample_rate))
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    time = (start_sample + np.arange(sample_count)) / config.sample_rate
    samples = np.zeros(
        (len(receiver_positions), sample_count), dtype=np.complex128
    )
    for source in sources:
        distances = np.linalg.norm(
            receiver_positions - source.position[None, :], axis=1
        )
        retarded_time = (
            time[None, :]
            - distances[:, None] / config.propagation_speed
            + source.symbol_offset_s
        )
        packet = packet_bits(source)
        symbol_indices = np.floor(retarded_time * config.symbol_rate).astype(np.int64)
        bits = packet[symbol_indices % len(packet)]
        envelope = np.where(bits == 1, config.level_one, config.level_zero)
        carrier = np.exp(
            1j * (2.0 * np.pi * source.carrier_hz * retarded_time + source.phase_rad)
        )
        samples += (
            source.amplitude
            * envelope
            * carrier
            / np.maximum(distances[:, None], np.finfo(float).eps)
        )

    noise_scale = config.noise_std / np.sqrt(2.0)
    noise = noise_scale * (
        rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
    )
    return samples + noise

