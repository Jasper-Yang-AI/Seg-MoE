import numpy as np


def test_combiner_shapes_smoke():
    N, K, M = 50, 9, 4
    probs = np.random.RandomState(0).rand(N, K, M).astype(np.float32)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    target = np.random.RandomState(1).randint(0, M, size=(N,)).astype(np.int64)

    from seg_moe.combiners.ole import OLECombiner

    ole = OLECombiner(mode="sgd_conv1x1", max_iter=10, lr=1e-2, seed=0)
    ole.fit(probs, target, num_classes=M)
    fused = ole.predict(probs)
    assert fused.shape == (N, M)

    from seg_moe.combiners.decision_template import DecisionTemplateCombiner

    dt = DecisionTemplateCombiner()
    dt.fit(probs, target, num_classes=M)
    pred = dt.predict(probs)
    assert pred.shape == (N,)

    from seg_moe.combiners.we_clpso import WECLPSOCombiner

    pso = WECLPSOCombiner(n_particles=5, iters=5, seed=0)
    pso.fit(probs, target, num_classes=M)
    fused2 = pso.predict(probs)
    assert fused2.shape == (N, M)
