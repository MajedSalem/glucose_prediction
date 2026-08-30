import numpy as np
import pytest

from glucose_prediction.evaluation.clarke import clarke_zones, zone_percentages, ZONES


@pytest.mark.parametrize("x, y, expected", [
    (100, 100, "A"),
    (300, 100, "D"),
    (150, 280, "C"),
    (150, 25, "C"),
    (70, 70, "A"),
])
def test_sanity_points(x, y, expected):
    assert clarke_zones([x], [y])[0] == expected


@pytest.mark.parametrize("x, y, expected", [
    (70, 180, "E"), (69.999, 180, "E"), (70, 180.001, "E"),
    (70, 179.999, "B"),
    (69.999, 100, "D"), (70, 100, "B"),
    (240, 150, "D"), (239.999, 150, "B"),
    (290, 400, "C"), (290.001, 400, "B"),
])
def test_boundary_strictness_is_self_consistent(x, y, expected):
    assert clarke_zones([x], [y])[0] == expected


@pytest.mark.parametrize("x, y", [(180, 70), (240, 70), (239.999, 70)])
def test_check_order_e_beats_c_and_d_at_shared_boundary(x, y):
    assert clarke_zones([x], [y])[0] == "E"


def test_vectorized_matches_naive_reference():
    def reference(x, y):
        if (x < 70 and y < 70) or (0.8 * x <= y <= 1.2 * x):
            return "A"
        if (x <= 70 and y >= 180) or (x >= 180 and y <= 70):
            return "E"
        if (70 <= x <= 290 and y >= x + 110) or (130 <= x <= 180 and y <= 1.4 * x - 182):
            return "C"
        if (x >= 240 and 70 <= y <= 180) or (x < 70 and 70 <= y <= 180):
            return "D"
        return "B"

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 400, 2000)
    y = rng.uniform(0, 400, 2000)
    expected = np.array([reference(xi, yi) for xi, yi in zip(x, y)])
    assert (clarke_zones(x, y) == expected).all()


def test_zone_percentages_match_reference_distribution_on_uniform_random():
    rng = np.random.default_rng(42)
    n = 200_000
    zones = clarke_zones(rng.uniform(0, 400, n), rng.uniform(0, 400, n))
    pct = zone_percentages(zones)
    reference = {"zone_A_pct": 20.8, "zone_B_pct": 27.9, "zone_C_pct": 16.2,
                 "zone_D_pct": 15.8, "zone_E_pct": 19.3}
    for key, expected in reference.items():
        assert pct[key] == pytest.approx(expected, abs=1.0)


def test_zone_percentages_sum_to_100():
    zones = clarke_zones([100, 300, 150], [100, 100, 280])
    assert sum(zone_percentages(zones).values()) == pytest.approx(100.0)
