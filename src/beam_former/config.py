"""Configuration for the acoustic beacon receiver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_PREAMBLE = np.array(
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8
)


@dataclass(frozen=True, slots=True)
class SignalConfig:
    sample_rate: float = 8_000.0
    propagation_speed: float = 343.0
    symbol_rate: float = 100.0
    payload_length: int = 32
    level_zero: float = 0.35
    level_one: float = 1.0
    noise_std: float = 0.012
    carrier_min_hz: float = 700.0
    carrier_max_hz: float = 2_200.0
    carrier_separation_hz: float = 300.0
    duration_s: float = 6.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.propagation_speed <= 0:
            raise ValueError("sample rate and propagation speed must be positive")
        if self.symbol_rate <= 0 or self.sample_rate / self.symbol_rate < 8:
            raise ValueError("symbol rate must allow at least eight samples per symbol")
        if self.payload_length < 1:
            raise ValueError("payload length must be positive")
        if not 0 <= self.level_zero < self.level_one:
            raise ValueError("ASK levels must satisfy 0 <= level_zero < level_one")
        if self.noise_std < 0:
            raise ValueError("noise standard deviation cannot be negative")
        if not 0 < self.carrier_min_hz < self.carrier_max_hz < self.sample_rate / 2:
            raise ValueError("carrier band must lie inside the Nyquist interval")
        if self.carrier_separation_hz <= 2 * self.symbol_rate:
            raise ValueError("carrier separation must exceed twice the symbol rate")
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")

    @property
    def samples_per_symbol(self) -> int:
        return int(round(self.sample_rate / self.symbol_rate))

    @property
    def packet_length(self) -> int:
        return len(DEFAULT_PREAMBLE) + self.payload_length


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    fft_size: int = 1_024
    fft_hop: int = 512
    max_candidates: int = 6
    peak_guard_hz: float = 180.0
    minimum_snr_db: float = 12.0
    covariance_shrinkage: float = 0.08
    diagonal_loading: float = 0.02
    azimuth_step_deg: float = 6.0
    elevation_step_deg: float = 6.0
    range_min_m: float = 0.8
    range_max_m: float = 3.8
    range_step_m: float = 0.25
    refine_angle_span_deg: float = 7.0
    refine_angle_step_deg: float = 1.0
    refine_range_span_m: float = 0.3
    refine_range_step_m: float = 0.05
    minimum_preamble_score: float = 0.68

    def __post_init__(self) -> None:
        if self.fft_size < 64 or self.fft_size & (self.fft_size - 1):
            raise ValueError("fft_size must be a power of two of at least 64")
        if not 0 < self.fft_hop <= self.fft_size:
            raise ValueError("fft_hop must be in (0, fft_size]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not 0 <= self.covariance_shrinkage < 1:
            raise ValueError("covariance shrinkage must be in [0, 1)")
        if self.diagonal_loading <= 0:
            raise ValueError("diagonal loading must be positive")
        if not 0 < self.range_min_m < self.range_max_m:
            raise ValueError("invalid scan range")
        if self.range_step_m <= 0:
            raise ValueError("range step must be positive")

