"""Windowed spectral detection and covariance estimation."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig, SignalConfig
from .models import CarrierDetection, ComplexArray, FloatArray


def stft_snapshots(
    samples: ComplexArray, fft_size: int, hop: int
) -> ComplexArray:
    if samples.ndim != 2 or samples.shape[1] < fft_size:
        raise ValueError("samples must have shape (receivers, time >= fft_size)")
    starts = np.arange(0, samples.shape[1] - fft_size + 1, hop)
    window = np.hanning(fft_size)
    snapshots = np.empty(
        (len(starts), samples.shape[0], fft_size), dtype=np.complex128
    )
    for index, start in enumerate(starts):
        snapshots[index] = np.fft.fft(
            samples[:, start : start + fft_size] * window[None, :], axis=1
        )
    return snapshots


def averaged_positive_spectrum(
    snapshots: ComplexArray, sample_rate: float
) -> tuple[FloatArray, FloatArray]:
    fft_size = snapshots.shape[-1]
    frequencies = np.fft.fftfreq(fft_size, 1.0 / sample_rate)
    positive = frequencies >= 0.0
    power = np.mean(np.abs(snapshots[:, :, positive]) ** 2, axis=(0, 1))
    return frequencies[positive], power.astype(np.float64)


def _interpolated_frequency(
    power: FloatArray, bin_index: int, sample_rate: float, fft_size: int
) -> float:
    if bin_index <= 0 or bin_index >= len(power) - 1:
        return bin_index * sample_rate / fft_size
    values = np.log(np.maximum(power[bin_index - 1 : bin_index + 2], 1e-30))
    denominator = values[0] - 2.0 * values[1] + values[2]
    offset = (
        0.0
        if abs(denominator) < 1e-20
        else 0.5 * (values[0] - values[2]) / denominator
    )
    offset = float(np.clip(offset, -0.5, 0.5))
    return (bin_index + offset) * sample_rate / fft_size


def detect_carriers(
    frequencies: FloatArray,
    power: FloatArray,
    signal_config: SignalConfig,
    analysis_config: AnalysisConfig,
) -> tuple[CarrierDetection, ...]:
    in_band = (frequencies >= signal_config.carrier_min_hz) & (
        frequencies <= signal_config.carrier_max_hz
    )
    band_indices = np.flatnonzero(in_band)
    if len(band_indices) < 3:
        return ()
    noise_floor = float(np.median(power[band_indices])) + 1e-30
    local = band_indices[
        (power[band_indices] >= power[band_indices - 1])
        & (power[band_indices] > power[band_indices + 1])
    ]
    threshold = noise_floor * 10.0 ** (analysis_config.minimum_snr_db / 10.0)
    available = [int(index) for index in local if power[index] >= threshold]
    available.sort(key=lambda index: power[index], reverse=True)

    selected: list[CarrierDetection] = []
    for index in available:
        frequency = _interpolated_frequency(
            power, index, signal_config.sample_rate, analysis_config.fft_size
        )
        if any(
            abs(frequency - item.frequency_hz) < analysis_config.peak_guard_hz
            for item in selected
        ):
            continue
        selected.append(
            CarrierDetection(
                frequency_hz=frequency,
                bin_index=index,
                power=float(power[index]),
                snr_db=float(10.0 * np.log10(power[index] / noise_floor)),
            )
        )
        if len(selected) == analysis_config.max_candidates:
            break
    return tuple(selected)


def covariance_at_bin(
    snapshots: ComplexArray,
    bin_index: int,
    analysis_config: AnalysisConfig,
) -> ComplexArray:
    values = snapshots[:, :, bin_index]
    covariance = values.T @ values.conj() / len(values)
    element_count = covariance.shape[0]
    mean_power = float(np.real(np.trace(covariance)) / element_count)
    identity = np.eye(element_count, dtype=np.complex128)
    shrinkage = analysis_config.covariance_shrinkage
    covariance = (1.0 - shrinkage) * covariance + shrinkage * mean_power * identity
    covariance += analysis_config.diagonal_loading * mean_power * identity
    return covariance.astype(np.complex128)
