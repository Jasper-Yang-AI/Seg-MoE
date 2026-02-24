"""Ensemble fusion combiners: OLE, Decision Template, WE-CLPSO, Majority Voting."""
from seg_moe.combiners.decision_template import DecisionTemplateCombiner
from seg_moe.combiners.majority_voting import MajorityVotingCombiner
from seg_moe.combiners.ole import OLECombiner
from seg_moe.combiners.we_clpso import WECLPSOCombiner

__all__ = [
    "OLECombiner",
    "DecisionTemplateCombiner",
    "WECLPSOCombiner",
    "MajorityVotingCombiner",
]
