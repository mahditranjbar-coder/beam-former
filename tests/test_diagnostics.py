import numpy as np

from beam_former.diagnostics import align_repeated_pattern


def test_alignment_finds_shift_and_reports_consensus_errors() -> None:
    tx = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    offset = 3
    indices = np.arange(5 * len(tx))
    rx = tx[(indices - offset) % len(tx)].copy()
    rx[1] ^= 1
    rx[19] ^= 1

    result = align_repeated_pattern(rx, tx)

    assert result.offset_bits == offset
    assert result.correlations[offset] == np.max(result.correlations)
    assert result.correlations[offset] == 0.9
    assert np.array_equal(result.rx_consensus, tx)
    assert result.bit_error_rate == 0.0


def test_alignment_rejects_empty_input() -> None:
    with np.testing.assert_raises(ValueError):
        align_repeated_pattern(
            np.empty(0, dtype=np.int8), np.array([0, 1], dtype=np.int8)
        )
