"""Scenario construction and truth-only validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AnalysisConfig, SignalConfig
from .geometry import angular_error_deg, circular_array
from .models import AnalysisResult, BeaconSource, ComplexArray, FloatArray
from .receiver import BeaconReceiver
from .signal import generate_sources, simulate_array_samples


@dataclass(frozen=True, slots=True)
class Scenario:
    receiver_positions: FloatArray
    sources: tuple[BeaconSource, ...]
    samples: ComplexArray
    signal_config: SignalConfig
    analysis_config: AnalysisConfig


@dataclass(frozen=True, slots=True)
class ValidationRow:
    estimated_frequency_hz: float
    true_frequency_hz: float
    frequency_error_hz: float
    angular_error_deg: float
    range_error_m: float
    locked: bool
    bit_error_rate: float | None


def create_scenario(
    *,
    seed: int = 7,
    source_count: int = 4,
    element_count: int = 16,
    array_radius_m: float = 0.18,
    signal_config: SignalConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> Scenario:
    signal = signal_config or SignalConfig()
    analysis = analysis_config or AnalysisConfig()
    rng = np.random.default_rng(seed)
    receivers = circular_array(element_count, array_radius_m)
    sources = generate_sources(rng, source_count, signal)
    samples = simulate_array_samples(
        receivers, sources, signal, rng=rng
    )
    return Scenario(receivers, sources, samples, signal, analysis)


def run_receiver(scenario: Scenario) -> AnalysisResult:
    receiver = BeaconReceiver(
        scenario.receiver_positions,
        scenario.signal_config,
        scenario.analysis_config,
    )
    return receiver.analyze_and_decode(scenario.samples)


def validate_against_truth(
    scenario: Scenario, analysis: AnalysisResult
) -> tuple[ValidationRow, ...]:
    rows: list[ValidationRow] = []
    for candidate in analysis.candidates:
        source = min(
            scenario.sources,
            key=lambda item: abs(
                item.carrier_hz - candidate.detection.frequency_hz
            ),
        )
        decode = candidate.decode
        bit_error_rate = None
        if decode is not None and decode.payload is not None:
            bit_error_rate = float(np.mean(decode.payload != source.payload))
        rows.append(
            ValidationRow(
                estimated_frequency_hz=candidate.detection.frequency_hz,
                true_frequency_hz=source.carrier_hz,
                frequency_error_hz=abs(
                    candidate.detection.frequency_hz - source.carrier_hz
                ),
                angular_error_deg=angular_error_deg(
                    candidate.localization.position, source.position
                ),
                range_error_m=abs(
                    candidate.localization.range_m - np.linalg.norm(source.position)
                ),
                locked=bool(decode and decode.locked),
                bit_error_rate=bit_error_rate,
            )
        )
    return tuple(rows)

