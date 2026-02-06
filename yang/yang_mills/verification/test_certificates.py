import pytest
import os
import re

def parse_latex_macros(tex_path):
    """
    Parses a LaTeX file and extracts \newcommand definitions into a dictionary.
    Returns dict: {macro_name: value_string}
    Example: \newcommand{\VerDobrushinNorm}{0.9375} -> {'VerDobrushinNorm': '0.9375'}
    """
    if not os.path.exists(tex_path):
        return {}
    
    macros = {}
    with open(tex_path, 'r') as f:
        content = f.read()
        # Regex to capture \newcommand{\Name}{Value}
        # Handles newlines and spacing vaguely, but specific export format is consistent
        matches = re.findall(r'\\newcommand\{\\(\w+)\}\{(.+?)\}', content)
        for name, val in matches:
            macros[name] = val
    return macros

class TestCertificates:
    """
    Regression tests for the exported verification certificates.
    Ensures that the rigorous proof artifacts (LaTeX macros) are present and correct.
    """
    
    @pytest.fixture
    def macros(self):
        # Locate the verification_results.tex file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tex_path = os.path.join(base_dir, "..", "single", "verification_results.tex")
        assert os.path.exists(tex_path), f"Certificate file not found at {tex_path}"
        return parse_latex_macros(tex_path)

    def test_dobrushin_certificate(self, macros):
        """Verify the Dobrushin Strong Coupling Handshake certificate."""
        assert "VerDobrushinStatus" in macros
        assert macros["VerDobrushinStatus"] == "PASS"
        assert "VerDobrushinNorm" in macros
        norm = float(macros["VerDobrushinNorm"])
        assert norm < 1.0, f"Dobrushin norm {norm} is not contractive (< 1.0)"
        assert macros["VerDobrushinBeta"] == "0.25"

    def test_interval_uniform_certificate(self, macros):
        """Verify the Interval-Uniform (Continuous Domain) certificate."""
        # This is the key "Idea #1" implementation check
        assert "VerIntervalStatus" in macros
        assert macros["VerIntervalStatus"] == "PASS", "Interval verification failed or skipped"
        
        assert "VerIntervalBallCount" in macros
        ball_count = int(macros["VerIntervalBallCount"])
        assert ball_count > 0, "No covering balls generated"
        
        assert "VerIntervalMaxJ" in macros
        max_j = float(macros["VerIntervalMaxJ"])
        assert max_j < 0.99, f"Max Jacobian {max_j} is not sufficiently contractive (< 0.99)"
        
        assert "VerIntervalMin" in macros, "Missing lower domain bound"
        assert "VerIntervalMax" in macros, "Missing upper domain bound"
        assert float(macros["VerIntervalMin"]) <= 0.25
        assert float(macros["VerIntervalMax"]) >= 6.0

    def test_pollution_constant_certificate(self, macros):
        """Verify the Ab-Initio Pollution Constant certificate."""
        assert "VerCPollMax" in macros
        c_poll = float(macros["VerCPollMax"])
        # We expect it to be small, definitely < 0.1 for the proof to hold
        assert c_poll < 0.1, f"Pollution constant {c_poll} is suspiciously large"
        assert c_poll > 0.0, "Pollution constant must be positive"

    def test_point_checks_consistency(self, macros):
        """Verify that point checks are also passing."""
        assert "VerTotalPoints" in macros
        assert macros["VerPassedPoints"] == macros["VerTotalPoints"], "Not all discrete points passed"
