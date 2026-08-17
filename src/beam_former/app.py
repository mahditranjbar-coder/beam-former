"""Interactive Matplotlib dashboard and command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from .config import DEFAULT_PREAMBLE
from .diagnostics import align_repeated_pattern
from .geometry import angles_from_position
from .scenario import create_scenario, run_receiver, validate_against_truth


def bits_to_text(bits: np.ndarray | None) -> str:
    if bits is None:
        return "(not decoded)"
    return "".join(str(int(bit)) for bit in bits)


def print_report(seed: int, source_count: int) -> int:
    scenario = create_scenario(seed=seed, source_count=source_count)
    analysis = run_receiver(scenario)
    rows = validate_against_truth(scenario, analysis)
    print(f"detected {len(rows)} carrier(s)")
    print("frequency   Δf      angle error   range error   lock   BER")
    for row in rows:
        ber = "---" if row.bit_error_rate is None else f"{row.bit_error_rate:.4f}"
        print(
            f"{row.estimated_frequency_hz:8.1f}  "
            f"{row.frequency_error_hz:6.2f}  "
            f"{row.angular_error_deg:11.2f}°  "
            f"{row.range_error_m:10.2f} m  "
            f"{str(row.locked):5s}  {ber}"
        )
    return 0


def show_dashboard(seed: int, source_count: int, show_truth: bool = True) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    scenario = create_scenario(seed=seed, source_count=source_count)
    analysis = run_receiver(scenario)
    validation = validate_against_truth(scenario, analysis)
    if not analysis.candidates:
        raise RuntimeError("no carriers were detected")

    figure = plt.figure(figsize=(15, 10))
    grid = figure.add_gridspec(
        3, 2, width_ratios=(1.05, 1.45), height_ratios=(1.0, 1.0, 0.9)
    )
    scene_axis = figure.add_subplot(grid[:2, 0], projection="3d")
    spectrum_axis = figure.add_subplot(grid[0, 1])
    envelope_axis = figure.add_subplot(grid[1, 1])
    diagnostic_grid = grid[2, :].subgridspec(1, 2, width_ratios=(0.8, 1.7))
    correlation_axis = figure.add_subplot(diagnostic_grid[0, 0])
    bits_axis = figure.add_subplot(diagnostic_grid[0, 1])
    figure.subplots_adjust(left=0.05, right=0.81, hspace=0.38, wspace=0.28)

    receivers = scenario.receiver_positions
    scene_axis.scatter(
        receivers[:, 0], receivers[:, 1], receivers[:, 2], label="array", s=28
    )
    estimates = np.stack(
        [candidate.localization.position for candidate in analysis.candidates]
    )
    scene_axis.scatter(
        estimates[:, 0],
        estimates[:, 1],
        estimates[:, 2],
        marker="x",
        s=85,
        label="estimated",
    )
    if show_truth:
        truth = np.stack([source.position for source in scenario.sources])
        scene_axis.scatter(
            truth[:, 0], truth[:, 1], truth[:, 2], marker="^", s=65, label="truth"
        )
    selected_line = scene_axis.plot([0, 0], [0, 0], [0, 0], linewidth=2.5)[0]
    scene_axis.set(
        title="Near-field 3D localization",
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
    )
    scene_axis.legend(loc="upper left")

    frequencies = analysis.frequencies_hz
    spectrum_db = 10.0 * np.log10(analysis.spectrum + 1e-30)
    spectrum_axis.plot(frequencies, spectrum_db, color="#315d9b")
    spectrum_axis.set_xlim(
        scenario.signal_config.carrier_min_hz - 100,
        scenario.signal_config.carrier_max_hz + 100,
    )
    spectrum_axis.set(
        title="Receiver spectrum and detected carriers",
        xlabel="frequency (Hz)",
        ylabel="power (dB)",
    )
    spectrum_axis.grid(alpha=0.25)
    for candidate in analysis.candidates:
        spectrum_axis.axvline(
            candidate.detection.frequency_hz, color="#d85b38", alpha=0.35
        )
    selected_frequency = spectrum_axis.axvline(0.0, color="black", linewidth=2)

    envelope_line = envelope_axis.plot([], [], color="#7f3c8d")[0]
    metrics_line = envelope_axis.plot([], [], ".", color="#e68310", markersize=3)[0]
    envelope_axis.set(
        title="Selected beam: isolated ASK envelope and recovered symbols",
        xlabel="time (s)",
        ylabel="amplitude",
    )
    envelope_axis.grid(alpha=0.25)

    correlation_axis.set(
        title="Raw RX / known TX cyclic correlation",
        xlabel="candidate offset (bits)",
        ylabel="correlation",
        ylim=(-1.05, 1.05),
    )
    correlation_axis.grid(alpha=0.25)
    bits_axis.set(
        title="Known TX pattern vs aligned RX consensus",
        xlabel="bit index: preamble then payload",
        ylabel="bit value",
        ylim=(-0.35, 1.65),
    )
    bits_axis.grid(alpha=0.25)

    labels = [
        f"{index}: {candidate.detection.frequency_hz:.1f} Hz"
        for index, candidate in enumerate(analysis.candidates)
    ]
    radio_axis = figure.add_axes((0.825, 0.57, 0.16, 0.25))
    radio_axis.set_title("Detected beacons")
    radio = RadioButtons(radio_axis, labels)
    info = figure.text(0.825, 0.51, "", va="top", family="monospace", fontsize=9)

    def select(label: str) -> None:
        index = labels.index(label)
        candidate = analysis.candidates[index]
        estimate = candidate.localization
        decode = candidate.decode
        position = estimate.position
        selected_line.set_data_3d(
            [0.0, position[0]], [0.0, position[1]], [0.0, position[2]]
        )
        selected_frequency.set_xdata([candidate.detection.frequency_hz] * 2)

        if decode is not None:
            decimation = max(1, len(decode.envelope) // 5_000)
            time = (
                np.arange(0, len(decode.envelope), decimation)
                / scenario.signal_config.sample_rate
            )
            envelope_line.set_data(time, decode.envelope[::decimation])
            if decode.symbol_phase_samples is not None:
                symbol_time = (
                    decode.symbol_phase_samples
                    + np.arange(len(decode.symbol_metrics))
                    * scenario.signal_config.samples_per_symbol
                ) / scenario.signal_config.sample_rate
                metrics_line.set_data(symbol_time, decode.symbol_metrics)
            envelope_axis.relim()
            envelope_axis.autoscale_view()

        azimuth, elevation, _ = angles_from_position(position)
        row = validation[index]
        payload_text = bits_to_text(decode.payload if decode else None)
        alignment = None
        if show_truth and decode is not None and len(decode.hard_bits):
            source = min(
                scenario.sources,
                key=lambda item: abs(
                    item.carrier_hz - candidate.detection.frequency_hz
                ),
            )
            tx_pattern = np.concatenate((DEFAULT_PREAMBLE, source.payload))
            alignment = align_repeated_pattern(decode.hard_bits, tx_pattern)

        correlation_axis.clear()
        bits_axis.clear()
        correlation_axis.set(
            title="Raw RX / known TX cyclic correlation",
            xlabel="candidate offset (bits)",
            ylabel="correlation",
            ylim=(-1.05, 1.05),
        )
        bits_axis.set(
            title="Known TX pattern vs aligned RX consensus",
            xlabel="bit index: preamble then payload",
            ylabel="bit value",
            ylim=(-0.35, 1.65),
        )
        correlation_axis.grid(alpha=0.25)
        bits_axis.grid(alpha=0.25)
        if alignment is not None:
            lags = np.arange(len(alignment.correlations))
            correlation_axis.plot(lags, alignment.correlations, ".-", markersize=3)
            correlation_axis.axvline(
                alignment.offset_bits, color="#d85b38", linestyle="--"
            )
            bit_indices = np.arange(len(alignment.tx_pattern))
            bits_axis.step(
                bit_indices,
                alignment.tx_pattern + 0.18,
                where="mid",
                label="known TX (+0.18)",
                color="#315d9b",
            )
            bits_axis.step(
                bit_indices,
                alignment.rx_consensus - 0.18,
                where="mid",
                label="aligned RX (-0.18)",
                color="#e68310",
            )
            mismatches = alignment.tx_pattern != alignment.rx_consensus
            bits_axis.plot(
                bit_indices[mismatches],
                np.full(np.sum(mismatches), 1.42),
                "x",
                color="#d62728",
                label="mismatch",
            )
            bits_axis.axvline(
                len(DEFAULT_PREAMBLE) - 0.5,
                color="black",
                alpha=0.35,
                linestyle=":",
            )
            bits_axis.legend(loc="upper right", ncols=3, fontsize=8)
        else:
            correlation_axis.text(
                0.5,
                0.5,
                "truth comparison hidden or no recovered bits",
                ha="center",
                va="center",
                transform=correlation_axis.transAxes,
            )
        lines = [
            f"frequency: {candidate.detection.frequency_hz:8.2f} Hz",
            f"SNR:       {candidate.detection.snr_db:8.1f} dB",
            f"azimuth:   {np.degrees(azimuth):8.2f}°",
            f"elevation: {np.degrees(elevation):8.2f}°",
            f"range:     {estimate.range_m:8.2f} m",
            "",
            f"locked:    {bool(decode and decode.locked)}",
            f"score:     {decode.preamble_score if decode else 0.0:8.3f}",
            f"packets:   {decode.packet_count if decode else 0:8d}",
            f"payload:   {payload_text[:16]}",
            f"           {payload_text[16:]}",
        ]
        if show_truth:
            bit_error_rate = (
                row.bit_error_rate if row.bit_error_rate is not None else np.nan
            )
            lines.extend(
                (
                    "",
                    f"angle err:{row.angular_error_deg:8.2f}°",
                    f"range err:{row.range_error_m:8.2f} m",
                    f"BER:      {bit_error_rate:8.4f}",
                )
            )
            if alignment is not None:
                lines.extend(
                    (
                        f"raw bits: {len(decode.hard_bits):8d}",
                        f"bit shift:{alignment.offset_bits:8d}",
                        f"max corr: {np.max(alignment.correlations):8.3f}",
                        f"align BER:{alignment.bit_error_rate:8.4f}",
                    )
                )
        info.set_text("\n".join(lines))
        figure.canvas.draw_idle()

    radio.on_clicked(select)
    select(labels[0])
    figure.suptitle(
        "Multi-frequency acoustic beacon receiver — detection → localization → "
        "beamforming → packet decode"
    )
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument(
        "--headless", action="store_true", help="print results without opening a GUI"
    )
    parser.add_argument(
        "--hide-truth",
        action="store_true",
        help="hide validation markers and errors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.headless:
        return print_report(arguments.seed, arguments.sources)
    show_dashboard(arguments.seed, arguments.sources, not arguments.hide_truth)
    return 0
