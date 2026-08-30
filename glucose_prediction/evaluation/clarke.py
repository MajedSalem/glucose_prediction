"""Vectorized Clarke Error Grid zone assignment (Clarke et al., 1987).

x = reference/true glucose, y = predicted glucose (mg/dL). Zones are checked
in strict priority order A, E, C, D, B -- some raw inequalities overlap
(e.g. x=180,y=70 satisfies both the C and E formulas), so earlier zone
assignments must never be overwritten by later checks.
"""
import numpy as np

ZONES = ("A", "B", "C", "D", "E")


def clarke_zones(y_true, y_pred) -> np.ndarray:
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    zone = np.full(x.shape, "", dtype="<U1")
    unassigned = np.ones(x.shape, dtype=bool)

    def assign(mask, label):
        hit = unassigned & mask
        zone[hit] = label
        unassigned[hit] = False

    assign(((x < 70) & (y < 70)) | ((y >= 0.8 * x) & (y <= 1.2 * x)), "A")
    assign(((x <= 70) & (y >= 180)) | ((x >= 180) & (y <= 70)), "E")
    assign(((x >= 70) & (x <= 290) & (y >= x + 110)) |
           ((x >= 130) & (x <= 180) & (y <= 1.4 * x - 182)), "C")
    assign(((x >= 240) & (y >= 70) & (y <= 180)) |
           ((x < 70) & (y >= 70) & (y <= 180)), "D")
    zone[unassigned] = "B"
    return zone


def zone_percentages(zones) -> dict:
    zones = np.asarray(zones)
    return {f"zone_{z}_pct": 100.0 * float(np.mean(zones == z)) for z in ZONES}
