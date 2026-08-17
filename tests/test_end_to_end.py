from beam_former.config import AnalysisConfig, SignalConfig
from beam_former.scenario import create_scenario, run_receiver, validate_against_truth


def test_three_beacons_are_detected_localized_and_decoded() -> None:
    signal = SignalConfig(duration_s=4.0, noise_std=0.008)
    analysis_config = AnalysisConfig(max_candidates=3, minimum_snr_db=12.0)
    scenario = create_scenario(
        seed=21,
        source_count=3,
        signal_config=signal,
        analysis_config=analysis_config,
    )

    result = run_receiver(scenario)
    validation = validate_against_truth(scenario, result)

    assert len(validation) == 3
    assert max(row.frequency_error_hz for row in validation) < 4.0
    assert max(row.angular_error_deg for row in validation) < 4.0
    # A single narrowband carrier gives strong angular information but only
    # coarse range curvature, particularly at the low end of the band.
    assert max(row.range_error_m for row in validation) < 0.7
    assert all(row.locked for row in validation)
    assert all(row.bit_error_rate == 0.0 for row in validation)
