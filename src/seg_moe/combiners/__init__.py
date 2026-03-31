"""Ensemble fusion combiners: OLE and Majority Voting."""
from seg_moe.combiners.majority_voting import MajorityVotingCombiner
from seg_moe.combiners.ole import OLECombiner

__all__ = [
    "OLECombiner",
    "MajorityVotingCombiner",
]
