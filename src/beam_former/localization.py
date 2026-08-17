"""Normalized near-field Capon localization and MVDR beamforming."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig, SignalConfig
from .geometry import angles_from_position, steering_vectors
from .models import ComplexArray, FloatArray, LocalizationEstimate


def _grid_points(
    azimuths: FloatArray, elevations: FloatArray, ranges: FloatArray
) -> FloatArray:
    azimuth, elevation, radius = np.meshgrid(
        azimuths, elevations, ranges, indexing="ij"
    )
    cosine = np.cos(elevation)
    return np.column_stack(
        (
            (radius * cosine * np.cos(azimuth)).ravel(),
            (radius * cosine * np.sin(azimuth)).ravel(),
            (radius * np.sin(elevation)).ravel(),
        )
    )


def _capon_scores(
    frequency_hz: float,
    points: FloatArray,
    covariance_inverse: ComplexArray,
    receiver_positions: FloatArray,
    propagation_speed: float,
) -> FloatArray:
    steering = steering_vectors(
        frequency_hz,
        points,
        receiver_positions,
        propagation_speed,
        normalize=True,
    )
    denominator = np.real(
        np.einsum(
            "gm,mn,gn->g",
            steering.conj(),
            covariance_inverse,
            steering,
            optimize=True,
        )
    )
    return 1.0 / np.maximum(denominator, 1e-30)


def localize_near_field(
    frequency_hz: float,
    covariance: ComplexArray,
    receiver_positions: FloatArray,
    signal_config: SignalConfig,
    analysis_config: AnalysisConfig,
) -> LocalizationEstimate:
    covariance_inverse = np.linalg.inv(covariance)
    azimuths = np.deg2rad(
        np.arange(0.0, 360.0, analysis_config.azimuth_step_deg)
    )
    elevations = np.deg2rad(
        np.arange(6.0, 90.0, analysis_config.elevation_step_deg)
    )
    ranges = np.arange(
        analysis_config.range_min_m,
        analysis_config.range_max_m + 0.5 * analysis_config.range_step_m,
        analysis_config.range_step_m,
    )
    coarse_points = _grid_points(azimuths, elevations, ranges)
    coarse_scores = _capon_scores(
        frequency_hz,
        coarse_points,
        covariance_inverse,
        receiver_positions,
        signal_config.propagation_speed,
    )
    coarse_best = coarse_points[int(np.argmax(coarse_scores))]
    azimuth, elevation, radius = angles_from_position(coarse_best)

    angle_span = np.deg2rad(analysis_config.refine_angle_span_deg)
    angle_step = np.deg2rad(analysis_config.refine_angle_step_deg)
    fine_azimuths = np.mod(
        np.arange(
            azimuth - angle_span,
            azimuth + angle_span + 0.5 * angle_step,
            angle_step,
        ),
        2.0 * np.pi,
    )
    fine_elevations = np.clip(
        np.arange(
            elevation - angle_span,
            elevation + angle_span + 0.5 * angle_step,
            angle_step,
        ),
        np.deg2rad(1.0),
        np.deg2rad(89.0),
    )
    fine_ranges = np.arange(
        max(analysis_config.range_min_m, radius - analysis_config.refine_range_span_m),
        min(analysis_config.range_max_m, radius + analysis_config.refine_range_span_m)
        + 0.5 * analysis_config.refine_range_step_m,
        analysis_config.refine_range_step_m,
    )
    fine_points = _grid_points(fine_azimuths, fine_elevations, fine_ranges)
    fine_scores = _capon_scores(
        frequency_hz,
        fine_points,
        covariance_inverse,
        receiver_positions,
        signal_config.propagation_speed,
    )
    best_index = int(np.argmax(fine_scores))
    position = fine_points[best_index]
    azimuth, elevation, radius = angles_from_position(position)
    return LocalizationEstimate(
        position=position,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        range_m=radius,
        score=float(fine_scores[best_index]),
    )


def mvdr_weights(
    frequency_hz: float,
    position: FloatArray,
    covariance: ComplexArray,
    receiver_positions: FloatArray,
    propagation_speed: float,
) -> ComplexArray:
    steering = steering_vectors(
        frequency_hz,
        position,
        receiver_positions,
        propagation_speed,
        normalize=False,
    )[0]
    solved = np.linalg.solve(covariance, steering)
    denominator = steering.conj() @ solved
    return (solved / denominator).astype(np.complex128)


def beamform(samples: ComplexArray, weights: ComplexArray) -> ComplexArray:
    return (weights.conj() @ samples).astype(np.complex128)
