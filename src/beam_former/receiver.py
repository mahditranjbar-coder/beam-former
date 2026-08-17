"""High-level local receiver pipeline."""

from __future__ import annotations

from dataclasses import replace

from .config import AnalysisConfig, SignalConfig
from .decoder import decode_ask_packets
from .localization import beamform, localize_near_field, mvdr_weights
from .models import AnalysisResult, CandidateResult, ComplexArray, FloatArray
from .spectral import (
    averaged_positive_spectrum,
    covariance_at_bin,
    detect_carriers,
    stft_snapshots,
)


class BeaconReceiver:
    """Discover, localize, select, and decode multi-frequency beacons."""

    def __init__(
        self,
        receiver_positions: FloatArray,
        signal_config: SignalConfig | None = None,
        analysis_config: AnalysisConfig | None = None,
    ) -> None:
        self.receiver_positions = receiver_positions
        self.signal_config = signal_config or SignalConfig()
        self.analysis_config = analysis_config or AnalysisConfig()

    def analyze(self, samples: ComplexArray) -> AnalysisResult:
        snapshots = stft_snapshots(
            samples,
            self.analysis_config.fft_size,
            self.analysis_config.fft_hop,
        )
        frequencies, spectrum = averaged_positive_spectrum(
            snapshots, self.signal_config.sample_rate
        )
        detections = detect_carriers(
            frequencies, spectrum, self.signal_config, self.analysis_config
        )
        candidates: list[CandidateResult] = []
        for detection in detections:
            covariance = covariance_at_bin(
                snapshots, detection.bin_index, self.analysis_config
            )
            localization = localize_near_field(
                detection.frequency_hz,
                covariance,
                self.receiver_positions,
                self.signal_config,
                self.analysis_config,
            )
            candidates.append(
                CandidateResult(
                    detection=detection,
                    localization=localization,
                    covariance=covariance,
                )
            )
        candidates.sort(key=lambda item: item.detection.frequency_hz)
        return AnalysisResult(
            frequencies_hz=frequencies,
            spectrum=spectrum,
            candidates=tuple(candidates),
        )

    def decode_candidate(
        self, samples: ComplexArray, candidate: CandidateResult
    ) -> CandidateResult:
        weights = mvdr_weights(
            candidate.detection.frequency_hz,
            candidate.localization.position,
            candidate.covariance,
            self.receiver_positions,
            self.signal_config.propagation_speed,
        )
        output = beamform(samples, weights)
        decode = decode_ask_packets(
            output,
            candidate.detection.frequency_hz,
            self.signal_config,
            self.analysis_config,
        )
        return replace(candidate, decode=decode)

    def analyze_and_decode(self, samples: ComplexArray) -> AnalysisResult:
        analysis = self.analyze(samples)
        decoded = tuple(
            self.decode_candidate(samples, candidate)
            for candidate in analysis.candidates
        )
        return replace(analysis, candidates=decoded)

