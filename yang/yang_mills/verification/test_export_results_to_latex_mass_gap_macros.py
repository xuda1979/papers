import os
import json


def _minimal_results():
    return {
        "metadata": {"generated": "TEST", "version": "1.0", "status": "PASS"},
        "regimes": {
            "strong_coupling_max": 0.25,
            "intermediate_min": 0.25,
            "intermediate_max": 6.0,
            "weak_coupling_min": 6.0,
        },
        "verification_points": {},
        "dobrushin_check": {"beta": 0.25, "norm_upper": 0.0, "status": "PASS"},
        "pollution_check": {"C_poll_upper_max": 0.0},
        "lsi_check": {"status": "PASS", "min_lsi_constant": 1e-6},
        "interval_check": {
            "covered_min": 0.25,
            "covered_max": 6.0,
            "ball_count": 0,
            "max_J_irr_continuous": 0.0,
            "status": "SKIPPED",
        },
        "summary": {
            "total_points": 0,
            "passed_points": 0,
            "max_J_irrelevant": 0.0,
            "min_J_irrelevant": 0.0,
            "contraction_margin": 0.0,
        },
    }


def test_export_includes_mass_gap_certificate_macros(tmp_path):
    # Import here so test discovery works even if some optional deps are missing.
    from export_results_to_latex import export_to_latex

    ver_dir = os.path.dirname(__file__)
    cert_path = os.path.join(ver_dir, "mass_gap_certificate.json")

    # Write a minimal-but-valid certificate file next to exporter (where it reads it).
    cert = {
        "schema": "yangmills.mass_gap_certificate.v1",
        "generated_by": "test",
        "claim": "ASSUMPTION-BASED",
        "clay_standard": False,
        "strict": False,
        "ok": True,
        "status": "CONDITIONAL",
        "reason": "theorem_boundary_or_mismatch",
        "mass_gap": {"kind": "dimensionless", "lower_bound": 1.234e-6, "derived_from": "lsi"},
        "os_audit": {"ok": True, "status": "CONDITIONAL", "reason": "theorem boundary"},
        "checkpoints": {"count": 0, "rows": []},
        "inputs": {},
    }

    wrote = False
    old_text = None
    if os.path.exists(cert_path):
        with open(cert_path, "r", encoding="utf-8") as f:
            old_text = f.read()

    try:
        with open(cert_path, "w", encoding="utf-8") as f:
            json.dump(cert, f)
        wrote = True

        out_tex = tmp_path / "verification_results.tex"
        export_to_latex(_minimal_results(), str(out_tex))

        text = out_tex.read_text(encoding="utf-8")
        assert "\\newcommand{\\VerMassGapCertPresent}{YES}" in text
        assert "\\newcommand{\\VerMassGapStatus}{CONDITIONAL}" in text
        assert "\\newcommand{\\VerOSAuditStatus}{CONDITIONAL}" in text
        assert "\\newcommand{\\VerMassGapLowerBound}{1.234000e-06}" in text
    finally:
        if wrote:
            if old_text is None:
                try:
                    os.remove(cert_path)
                except OSError:
                    pass
            else:
                with open(cert_path, "w", encoding="utf-8") as f:
                    f.write(old_text)
