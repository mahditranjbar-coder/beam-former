# Beam Former

A working simulation of a local multi-frequency acoustic beacon receiver. The
receiver is given only the complex samples measured by a circular array. It
must discover active carriers, estimate where they came from, form a beam
toward a selected beacon, recover symbol timing, find the packet preamble, and
decode the payload.

The default deterministic scene contains four independent ASK beacons. On the
reference seed, all four are detected within 0.06 Hz, localized within 0.5°,
and decoded with zero bit errors.

## Receiver pipeline

```text
array samples
    → windowed STFT and carrier detection
    → covariance estimation with shrinkage and diagonal loading
    → normalized near-field Capon scan (coarse then refined)
    → MVDR beam toward the selected beacon
    → complex downconversion and low-pass isolation
    → receiver-side symbol-phase search
    → soft preamble correlation and packet majority vote
```

No source timing, position, carrier, or payload is passed into the receiver.
Ground truth is kept in the scenario layer and is used only to report errors.

## Run the dashboard

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
beam-former
```

Use the radio buttons to select a detected beacon. The dashboard updates the
estimated 3D direction, spectrum cursor, isolated ASK envelope, recovered
symbol levels, preamble score, packet count, payload, and validation errors.
For simulated data it also retains the unaligned hard-decision RX stream and
shows two truth-referenced diagnostics:

- cyclic bipolar correlation versus every possible packet offset; the peak is
  the estimated bit shift (`+1` means exact agreement, `0` chance-level, and
  `-1` inversion), and
- the known transmitted `preamble + payload` overlaid with an aligned RX
  consensus formed by folding all recovered repetitions onto one packet.

Red crosses mark aligned bit mismatches. `align BER` is computed only after the
cyclic shift is estimated. These comparisons are dashboard validation only;
the transmitter pattern is not passed to carrier detection, localization,
timing recovery, or packet decoding.

To hide all simulator truth information:

```bash
beam-former --hide-truth
```

For a noninteractive receiver report:

```bash
beam-former --headless
```

Example output:

```text
detected 4 carrier(s)
frequency   Δf      angle error   range error   lock   BER
   707.9    0.05         0.25°        0.02 m  True   0.0000
  1037.8    0.05         0.16°        0.01 m  True   0.0000
  1637.6    0.03         0.33°        0.02 m  True   0.0000
  2045.8    0.05         0.44°        0.00 m  True   0.0000
```

## Project structure

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated signal and receiver settings |
| `signal.py` | Source generation and retarded-time array simulation |
| `spectral.py` | STFT, carrier detection, and covariance estimation |
| `localization.py` | Near-field Capon scan and MVDR beamforming |
| `decoder.py` | Carrier isolation, timing recovery, and packet decoding |
| `diagnostics.py` | Truth-referenced cyclic bit alignment and comparison |
| `receiver.py` | Receiver-only orchestration |
| `scenario.py` | Test scenes and truth-only validation |
| `app.py` | Interactive dashboard and command-line report |

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m build
```

The tests separately verify signal generation, carrier detection, near-field
localization, receiver-side symbol timing, payload decoding, and a complete
three-beacon run.

## Physical limits

This remains an educational narrowband model, not calibrated measurement
software. Direction is generally much better conditioned than range. A single
narrowband carrier may produce a coarse or ambiguous range estimate when the
source is outside the array's strong near-field region. Reliable range in a
real system would normally require a wider-band preamble, calibrated amplitude,
multiple apertures, or another ranging observable.

The default 16-element, 0.18 m-radius array was chosen to avoid the severe
spatial aliasing and covariance-rank problems in the original sketch while
keeping the demonstration fast.

## License

GNU General Public License v3.0 (`GPL-3.0-only`).
