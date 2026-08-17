"""Truth-referenced bit alignment diagnostics for simulated scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class BitAlignment:
    """Comparison of an unaligned recovered stream with a repeated TX pattern."""

    offset_bits: int
    correlations: FloatArray
    tx_pattern: IntArray
    rx_consensus: IntArray
    vote_confidence: FloatArray
    bit_error_rate: float
    compared_bits: int


def align_repeated_pattern(rx_bits: IntArray, tx_pattern: IntArray) -> BitAlignment:
    """Find the cyclic TX/RX offset and fold RX repetitions into one packet.

    ``offset_bits`` is the RX-stream index at which TX pattern bit zero occurs,
    modulo the packet length.  Bipolar correlation has an intuitive scale:
    +1 is exact agreement, 0 is chance-level agreement, and -1 is inversion.
    """
    rx = np.asarray(rx_bits, dtype=np.int8)
    tx = np.asarray(tx_pattern, dtype=np.int8)
    if rx.ndim != 1 or tx.ndim != 1:
        raise ValueError("RX bits and TX pattern must be one-dimensional")
    if len(rx) == 0 or len(tx) == 0:
        raise ValueError("RX bits and TX pattern must not be empty")
    if np.any((rx != 0) & (rx != 1)) or np.any((tx != 0) & (tx != 1)):
        raise ValueError("bit arrays may contain only zero and one")

    indices = np.arange(len(rx))
    rx_pm = 2.0 * rx.astype(np.float64) - 1.0
    tx_pm = 2.0 * tx.astype(np.float64) - 1.0
    correlations = np.asarray(
        [
            np.mean(rx_pm * tx_pm[(indices - offset) % len(tx)])
            for offset in range(len(tx))
        ],
        dtype=np.float64,
    )
    offset = int(np.argmax(correlations))

    consensus = np.empty(len(tx), dtype=np.int8)
    confidence = np.empty(len(tx), dtype=np.float64)
    observed = np.zeros(len(tx), dtype=bool)
    compared = 0
    for tx_index in range(len(tx)):
        samples = rx[(indices - offset) % len(tx) == tx_index]
        if len(samples) == 0:
            consensus[tx_index] = 0
            confidence[tx_index] = 0.0
            continue
        fraction_one = float(np.mean(samples))
        observed[tx_index] = True
        consensus[tx_index] = int(fraction_one >= 0.5)
        confidence[tx_index] = abs(2.0 * fraction_one - 1.0)
        compared += 1

    bit_error_rate = (
        float(np.mean(consensus[observed] != tx[observed]))
        if np.any(observed)
        else float("nan")
    )
    return BitAlignment(
        offset_bits=offset,
        correlations=correlations,
        tx_pattern=tx.copy(),
        rx_consensus=consensus,
        vote_confidence=confidence,
        bit_error_rate=bit_error_rate,
        compared_bits=compared,
    )
