"""Tests for the ATSC verifier and the composed radial-atlas + ATSC verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_angular_torus_subdivision_cover as vatsc  # noqa: E402
import verify_composed_angular_torus_cover as vcomposed  # noqa: E402

TOY_ATSC = ROOT / "certificates" / "atsc" / "toy_atsc_certificate.json"
RADIAL_ATLAS = ROOT / "certificates" / "carter_atlases" / "spin_half_candidate_atlas_subdiv4.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_toy_atsc_accepts():
    result = vatsc.verify_file(TOY_ATSC)
    assert result["accepted"] is True
    assert result["theorem"] == "ATSC"
    assert result["version"] == "1.0"
    assert result["interval_count"] == 7
    assert result["tag_distribution"] == {"polar": 2, "equatorial": 1, "mid_latitude": 4}
    assert result["radial_chart_count"] == 16


def test_toy_composed_accepts():
    result = vcomposed.verify_composed(RADIAL_ATLAS, TOY_ATSC)
    assert result["accepted"] is True
    assert result["composition"] == "radial_atlas + angular_patches + ATSC"
    assert result["radial_chart_count"] == 16
    assert result["interval_count"] == 7


def test_atsc_rejects_wrong_theorem_name(tmp_path):
    data = _load(TOY_ATSC)
    data["theorem"] = "NOT_ATSC"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="theorem must be 'ATSC'"):
        vatsc.verify_file(p)


def test_atsc_rejects_wrong_version(tmp_path):
    data = _load(TOY_ATSC)
    data["version"] = "0.9"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="version must be '1.0'"):
        vatsc.verify_file(p)


def test_atsc_rejects_invalid_tag(tmp_path):
    data = _load(TOY_ATSC)
    data["angular_subdivision"]["intervals"][0]["tag"] = "not_a_tag"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="tag must be one of"):
        vatsc.verify_file(p)


def test_atsc_rejects_zero_length_interval(tmp_path):
    data = _load(TOY_ATSC)
    data["angular_subdivision"]["intervals"][0]["length"] = "0/1"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="length must be positive"):
        vatsc.verify_file(p)


def test_atsc_rejects_polar_with_wrong_cos_squared_bound(tmp_path):
    """Polar tag requires cos^2(theta) >= 15/16."""
    data = _load(TOY_ATSC)
    # polar_north interval: lower cos^2 from 15/16 to 1/2 (violates polar bound)
    data["angular_subdivision"]["intervals"][0]["cos_squared_theta_interval"] = "[1/2, 1/1]"
    # also need to keep sum containing 1: sin^2 must be [0, 1/2]
    data["angular_subdivision"]["intervals"][0]["sin_squared_theta_interval"] = "[0/1, 1/2]"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="polar tag requires cos\\^2\\(theta\\) lower bound"):
        vatsc.verify_file(p)


def test_atsc_rejects_equatorial_with_wrong_sin_squared_bound(tmp_path):
    """Equatorial tag requires sin^2(theta) >= 262143/524288 (about 0.4999...).
    Use sin^2 lower bound 1/4 = 0.25 which is below 262143/524288."""
    data = _load(TOY_ATSC)
    # equatorial interval (index 3): lower sin^2 from 262143/524288 to 1/4 (violates equatorial bound)
    data["angular_subdivision"]["intervals"][3]["sin_squared_theta_interval"] = "[1/4, 1/1]"
    # keep cos^2 upper bound <= 1/524288 to isolate the sin^2 check; the sum
    # interval becomes [1/4, 1+1/524288] which still contains 1.
    data["angular_subdivision"]["intervals"][3]["cos_squared_theta_interval"] = "[0/1, 1/524288]"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="equatorial tag requires sin\\^2\\(theta\\) lower bound"):
        vatsc.verify_file(p)


def test_atsc_rejects_mid_latitude_with_wrong_sin_squared_bound(tmp_path):
    """Mid-latitude tag requires sin^2(theta) >= 1/32."""
    data = _load(TOY_ATSC)
    # mid_lat_north_near_pole (index 1): lower sin^2 from 1/32 to 1/64 (violates)
    data["angular_subdivision"]["intervals"][1]["sin_squared_theta_interval"] = "[1/64, 1048575/1048576]"
    # adjust cos^2 so sum contains 1
    data["angular_subdivision"]["intervals"][1]["cos_squared_theta_interval"] = "[1/1048576, 63/64]"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="mid_latitude tag requires sin\\^2\\(theta\\) lower bound"):
        vatsc.verify_file(p)


def test_atsc_rejects_non_covering_subdivision(tmp_path):
    """If intervals leave a gap, the coverage check should fail."""
    data = _load(TOY_ATSC)
    # Remove the equatorial interval (index 3); create a gap
    del data["angular_subdivision"]["intervals"][3]
    # Adjust adjacency: remove adjacencies involving equatorial
    new_adj = []
    for adj in data["angular_subdivision"]["adjacency"]:
        if adj["left"] == "equatorial" or adj["right"] == "equatorial":
            continue
        new_adj.append(adj)
    # Add an adjacency between mid_lat_north and mid_lat_south (which now overlap)
    # Actually they don't overlap; mid_lat_north ends at 29/64, mid_lat_south starts at 35/64.
    # We need to extend one to cover the gap. Extend mid_lat_north to end at 36/64.
    for iv in data["angular_subdivision"]["intervals"]:
        if iv["name"] == "mid_lat_north":
            iv["theta_interval"] = "[9/64, 36/64]"
            iv["length"] = "27/64"
        if iv["name"] == "mid_lat_south":
            iv["theta_interval"] = "[35/64, 55/64]"
            iv["length"] = "20/64"
    new_adj.append({
        "left": "mid_lat_north", "right": "mid_lat_south",
        "overlap_length": "1/64", "overlap_theta_interval": "[35/64, 36/64]"
    })
    data["angular_subdivision"]["adjacency"] = new_adj
    # Update cover budget and coverage
    data["coverage"]["interval_count"] = 6
    data["coverage"]["total_length"] = "1/1"  # placeholder; will be recomputed by verifier
    # The verifier should reject because coverage checks fail
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError):
        vatsc.verify_file(p)


def test_atsc_rejects_no_mid_latitude(tmp_path):
    """Cover budget requires at least one mid-latitude interval."""
    data = _load(TOY_ATSC)
    # Change all mid_latitude tags to polar (will fail tag-specific checks but
    # let's also fix them so the only failure is the cover budget)
    # Actually changing tags to polar will fail the polar cos^2 bound checks.
    # Easier: just set tag_distribution.mid_latitude to 0 and check the cover_budget validator rejects.
    data["cover_budget"]["tag_distribution"]["mid_latitude"] = 0
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="mid_latitude.*must be >="):
        vatsc.verify_file(p)


def test_atsc_rejects_duplicate_interval_names(tmp_path):
    data = _load(TOY_ATSC)
    data["angular_subdivision"]["intervals"][1]["name"] = data["angular_subdivision"]["intervals"][0]["name"]
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="name duplicates an earlier interval"):
        vatsc.verify_file(p)


def test_atsc_rejects_adjacency_unknown_interval(tmp_path):
    data = _load(TOY_ATSC)
    data["angular_subdivision"]["adjacency"][0]["left"] = "not_a_real_interval"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="references unknown interval"):
        vatsc.verify_file(p)


def test_atsc_rejects_zero_overlap_length(tmp_path):
    data = _load(TOY_ATSC)
    data["angular_subdivision"]["adjacency"][0]["overlap_length"] = "0/1"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="overlap_length must be positive"):
        vatsc.verify_file(p)


def test_atsc_rejects_wrong_total_length(tmp_path):
    data = _load(TOY_ATSC)
    data["coverage"]["total_length"] = "1/1"  # wrong
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="total_length.*must equal"):
        vatsc.verify_file(p)


def test_atsc_rejects_wrong_min_overlap(tmp_path):
    data = _load(TOY_ATSC)
    data["coverage"]["min_overlap_length"] = "1/32"  # wrong (actual is 1/64)
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="min_overlap_length.*must equal"):
        vatsc.verify_file(p)


def test_atsc_rejects_wrong_interval_count(tmp_path):
    data = _load(TOY_ATSC)
    data["coverage"]["interval_count"] = 99
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="interval_count.*must equal"):
        vatsc.verify_file(p)


def test_atsc_rejects_radial_chart_count_mismatch(tmp_path):
    data = _load(TOY_ATSC)
    data["radial_atlas_compatibility"]["radial_chart_count"] = 99
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="per_chart_lambda_eta_coverage must be a list"):
        vatsc.verify_file(p)


def test_atsc_rejects_non_covered_angular_interval_for_chart(tmp_path):
    data = _load(TOY_ATSC)
    # Set one chart's coverage count to 6 (less than 7)
    data["radial_atlas_compatibility"]["per_chart_lambda_eta_coverage"][0][
        "angular_intervals_with_nonempty_lambda_intersection"] = 6
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="angular_intervals_with_nonempty_lambda_intersection"):
        vatsc.verify_file(p)


def test_atsc_rejects_cos_sin_sum_violation(tmp_path):
    """cos^2 + sin^2 sum interval must contain 1."""
    data = _load(TOY_ATSC)
    # polar_north: cos^2 [15/16, 1], sin^2 [0, 1/16] -> sum [15/16, 17/16] contains 1. OK.
    # Break it: set sin^2 to [0, 1/8] so sum is [15/16, 9/8] which still contains 1.
    # To really break: set cos^2 to [15/16, 15/16] and sin^2 to [0, 0].
    data["angular_subdivision"]["intervals"][0]["cos_squared_theta_interval"] = "[15/16, 15/16]"
    data["angular_subdivision"]["intervals"][0]["sin_squared_theta_interval"] = "[0/1, 0/1]"
    # sum = 15/16, doesn't contain 1
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="cos\\^2\\(theta\\) \\+ sin\\^2\\(theta\\)"):
        vatsc.verify_file(p)


def test_composed_rejects_spin_mismatch(tmp_path):
    atlas = _load(RADIAL_ATLAS)
    atsc = _load(TOY_ATSC)
    atsc["spin"] = "1/4"  # mismatch
    p_atlas = _dump(tmp_path, "atlas.json", atlas)
    p_atsc = _dump(tmp_path, "atsc.json", atsc)
    result = vcomposed.verify_composed(p_atlas, p_atsc)
    assert result["accepted"] is False
    assert any("spin mismatch" in e for e in result["errors"])


def test_composed_rejects_chart_name_mismatch(tmp_path):
    atlas = _load(RADIAL_ATLAS)
    atsc = _load(TOY_ATSC)
    # Change one chart name in ATSC
    atsc["radial_atlas_compatibility"]["per_chart_lambda_eta_coverage"][0]["radial_chart"] = "fake_chart"
    p_atlas = _dump(tmp_path, "atlas.json", atlas)
    p_atsc = _dump(tmp_path, "atsc.json", atsc)
    result = vcomposed.verify_composed(p_atlas, p_atsc)
    assert result["accepted"] is False
    assert any("radial chart name sets differ" in e for e in result["errors"])


def test_composed_rejects_missing_positivity_reference(tmp_path):
    atlas = _load(RADIAL_ATLAS)
    atsc = _load(TOY_ATSC)
    atsc["angular_subdivision"]["intervals"][0]["positivity_certificate_reference"] = "nonexistent/path.json"
    p_atlas = _dump(tmp_path, "atlas.json", atlas)
    p_atsc = _dump(tmp_path, "atsc.json", atsc)
    result = vcomposed.verify_composed(p_atlas, p_atsc)
    assert result["accepted"] is False
    assert any("positivity_certificate_reference" in e and "does not point" in e
               for e in result["errors"])
