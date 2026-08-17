"""Array geometry and steering vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import FloatArray


def circular_array(element_count: int = 16, radius_m: float = 0.18) -> FloatArray:
    if element_count < 4:
        raise ValueError("a circular array needs at least four elements")
    if radius_m <= 0:
        raise ValueError("array radius must be positive")
    angles = np.linspace(0.0, 2.0 * np.pi, element_count, endpoint=False)
    return np.column_stack(
        (radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(element_count))
    )


def unit_vector(azimuth_rad: float, elevation_rad: float) -> FloatArray:
    cosine = np.cos(elevation_rad)
    return np.array(
        [
            cosine * np.cos(azimuth_rad),
            cosine * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )


def angles_from_position(position: FloatArray) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(position))
    if radius == 0:
        raise ValueError("position cannot be the origin")
    azimuth = float(np.arctan2(position[1], position[0]) % (2.0 * np.pi))
    elevation = float(np.arcsin(np.clip(position[2] / radius, -1.0, 1.0)))
    return azimuth, elevation, radius


def angular_error_deg(first: FloatArray, second: FloatArray) -> float:
    first_unit = first / np.linalg.norm(first)
    second_unit = second / np.linalg.norm(second)
    cosine = np.clip(float(first_unit @ second_unit), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def steering_vectors(
    frequency_hz: float,
    points: FloatArray,
    receiver_positions: FloatArray,
    propagation_speed: float,
    *,
    normalize: bool,
) -> NDArray[np.complex128]:
    points_2d = np.atleast_2d(points).astype(np.float64)
    distances = np.linalg.norm(
        points_2d[:, None, :] - receiver_positions[None, :, :], axis=2
    )
    distances = np.maximum(distances, np.finfo(float).eps)
    wave_number = 2.0 * np.pi * frequency_hz / propagation_speed
    steering = np.exp(-1j * wave_number * distances) / distances
    if normalize:
        steering /= np.linalg.norm(steering, axis=1, keepdims=True)
    return steering.astype(np.complex128)

