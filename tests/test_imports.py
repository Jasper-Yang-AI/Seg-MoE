def test_imports():
    import seg_moe  # noqa: F401

    from seg_moe.data.dataset_2d import SegmentationDataset2D  # noqa: F401
    from seg_moe.models.factory_2d import build_expert  # noqa: F401
    from seg_moe.combiners.ole import OLECombiner  # noqa: F401
    from seg_moe.combiners.decision_template import DecisionTemplateCombiner  # noqa: F401
    from seg_moe.combiners.we_clpso import WECLPSOCombiner  # noqa: F401
