"""Carrier isolation, symbol-timing recovery, and ASK packet decoding."""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_PREAMBLE, AnalysisConfig, SignalConfig
from .models import ComplexArray, DecodeResult, FloatArray


def lowpass_complex(
    samples: ComplexArray, sample_rate: float, cutoff_hz: float, taps: int = 161
) -> ComplexArray:
    if taps % 2 == 0:
        raise ValueError("FIR tap count must be odd")
    normalized = cutoff_hz / sample_rate
    indices = np.arange(taps) - taps // 2
    kernel = 2.0 * normalized * np.sinc(2.0 * normalized * indices)
    kernel *= np.hamming(taps)
    kernel /= np.sum(kernel)
    return np.convolve(samples, kernel, mode="same").astype(np.complex128)


def _symbol_metrics(
    envelope: FloatArray, phase: int, samples_per_symbol: int
) -> FloatArray:
    centers = phase + np.arange(
        0, max(0, len(envelope) - phase) // samples_per_symbol
    ) * samples_per_symbol
    half_width = max(1, int(0.28 * samples_per_symbol))
    valid = centers[(centers >= half_width) & (centers + half_width < len(envelope))]
    if len(valid) == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [np.mean(envelope[index - half_width : index + half_width]) for index in valid],
        dtype=np.float64,
    )


def _soft_symbols(metrics: FloatArray) -> tuple[FloatArray, float, float]:
    low = float(np.percentile(metrics, 20.0))
    high = float(np.percentile(metrics, 80.0))
    threshold = 0.5 * (low + high)
    scale = max(0.12 * (high - low), 1e-12)
    return np.tanh((metrics - threshold) / scale), low, high


def decode_ask_packets(
    beamformed: ComplexArray,
    carrier_hz: float,
    signal_config: SignalConfig,
    analysis_config: AnalysisConfig,
) -> DecodeResult:
    sample_rate = signal_config.sample_rate
    time = np.arange(len(beamformed)) / sample_rate
    mixed = beamformed * np.exp(-1j * 2.0 * np.pi * carrier_hz * time)
    cutoff = min(
        1.35 * signal_config.symbol_rate,
        0.44 * signal_config.carrier_separation_hz,
    )
    baseband = lowpass_complex(mixed, sample_rate, cutoff)
    envelope = np.abs(baseband).astype(np.float64)
    samples_per_symbol = signal_config.samples_per_symbol
    preamble_pm = 2.0 * DEFAULT_PREAMBLE.astype(float) - 1.0

    best_score = -1.0
    best_phase: int | None = None
    best_metrics = np.empty(0, dtype=np.float64)
    for phase in range(samples_per_symbol):
        metrics = _symbol_metrics(envelope, phase, samples_per_symbol)
        if len(metrics) < signal_config.packet_length:
            continue
        soft, _, _ = _soft_symbols(metrics)
        scores = np.correlate(soft, preamble_pm, mode="valid") / len(preamble_pm)
        score = float(np.max(scores))
        if score > best_score:
            best_score = score
            best_phase = phase
            best_metrics = metrics

    if best_phase is None or best_score < analysis_config.minimum_preamble_score:
        return DecodeResult(
            locked=False,
            payload=None,
            preamble_score=max(0.0, best_score),
            preamble_errors=None,
            symbol_phase_samples=best_phase,
            packet_count=0,
            level_zero=None,
            level_one=None,
            envelope=envelope,
            symbol_metrics=best_metrics,
        )

    soft, level_zero, level_one = _soft_symbols(best_metrics)
    scores = np.correlate(soft, preamble_pm, mode="valid") / len(preamble_pm)
    best_start = int(np.argmax(scores))
    threshold = 0.5 * (level_zero + level_one)
    hard_bits = (best_metrics > threshold).astype(np.int8)
    packet_length = signal_config.packet_length
    phase_class = best_start % packet_length
    payloads: list[np.ndarray] = []
    preamble_errors: list[int] = []
    for start in range(phase_class, len(hard_bits) - packet_length + 1, packet_length):
        packet = hard_bits[start : start + packet_length]
        errors = int(np.sum(packet[: len(DEFAULT_PREAMBLE)] != DEFAULT_PREAMBLE))
        if errors <= 2:
            preamble_errors.append(errors)
            payloads.append(packet[len(DEFAULT_PREAMBLE) :].copy())

    if not payloads:
        return DecodeResult(
            locked=False,
            payload=None,
            preamble_score=best_score,
            preamble_errors=None,
            symbol_phase_samples=best_phase,
            packet_count=0,
            level_zero=level_zero,
            level_one=level_one,
            envelope=envelope,
            symbol_metrics=best_metrics,
        )

    stacked = np.stack(payloads)
    payload = (np.mean(stacked, axis=0) >= 0.5).astype(np.int8)
    return DecodeResult(
        locked=True,
        payload=payload,
        preamble_score=best_score,
        preamble_errors=min(preamble_errors),
        symbol_phase_samples=best_phase,
        packet_count=len(payloads),
        level_zero=level_zero,
        level_one=level_one,
        envelope=envelope,
        symbol_metrics=best_metrics,
    )
