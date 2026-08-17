"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class BeaconSource:
    position: FloatArray
    carrier_hz: float
    payload: IntArray
    amplitude: float
    phase_rad: float
    symbol_offset_s: float


@dataclass(frozen=True, slots=True)
class CarrierDetection:
    frequency_hz: float
    bin_index: int
    power: float
    snr_db: float


@dataclass(frozen=True, slots=True)
class LocalizationEstimate:
    position: FloatArray
    azimuth_rad: float
    elevation_rad: float
    range_m: float
    score: float


@dataclass(frozen=True, slots=True)
class DecodeResult:
    locked: bool
    payload: IntArray | None
    preamble_score: float
    preamble_errors: int | None
    symbol_phase_samples: int | None
    packet_count: int
    level_zero: float | None
    level_one: float | None
    envelope: FloatArray
    symbol_metrics: FloatArray


@dataclass(frozen=True, slots=True)
class CandidateResult:
    detection: CarrierDetection
    localization: LocalizationEstimate
    covariance: ComplexArray
    decode: DecodeResult | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    frequencies_hz: FloatArray
    spectrum: FloatArray
    candidates: tuple[CandidateResult, ...]

