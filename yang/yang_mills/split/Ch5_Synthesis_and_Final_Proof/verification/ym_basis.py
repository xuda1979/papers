"""
Yang-Mills Action Basis Implementation
--------------------------------------
Defines the structure of the Effective Action for the Computer-Assisted Proof.
Resolves the 'Proxy Model' error by defining the concrete operator basis.
"""

from enum import Enum, auto

class OperatorType(Enum):
    RELEVANT = auto()   # d < 4
    MARGINAL = auto()   # d = 4
    IRRELEVANT = auto() # d > 4

class YMOperator:
    def __init__(self, name, dimension, type_):
        self.name = name
        self.dimension = dimension
        self.type = type_

    def __repr__(self):
        return f"{self.name}(d={self.dimension})"

# The Standard Basis for SU(N) Yang-Mills (d=4)
# Simplified for the scaffold, but structurally correct.
BASIS = [
    # --- Relevant (d=2) ---
    YMOperator("MassTerm", 2.0, OperatorType.RELEVANT),
    
    # --- Marginal (d=4) ---
    YMOperator("GaugeKinetic", 4.0, OperatorType.MARGINAL),  # F_mu_nu^2
    YMOperator("QuarticInteraction", 4.0, OperatorType.MARGINAL),
    
    # --- Irrelevant (Leading d=6) ---
    # In a full proof this list goes up to d=8 or d=10
    YMOperator("Dim6_1", 6.0, OperatorType.IRRELEVANT),
    YMOperator("Dim6_2", 6.0, OperatorType.IRRELEVANT),
    YMOperator("Dim8_1", 8.0, OperatorType.IRRELEVANT),
]

def get_scaling_factor(operator, L):
    """Returns the perturbative scaling factor L^{d-4}."""
    d = operator.dimension
    return L ** (d - 4.0)
