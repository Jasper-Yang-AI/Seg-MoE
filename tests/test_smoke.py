import numpy as np


def test_combiner_shapes_smoke():
    N, K, M = 50, 3, 4
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

    from seg_moe.combiners.ole import fit_from_oof, fuse, predict

    W_bvls = fit_from_oof(probs, target, method="bvls", bounds=(0.0, 1.0), seed=0)
    scores = fuse(probs, W_bvls)
    seg = predict(probs, W_bvls)
    assert scores.shape == (N, M)
    assert seg.shape == (N,)

    W_nnls = fit_from_oof(probs, target, method="nnls")
    scores2 = fuse(probs, W_nnls)
    seg2 = predict(probs, W_nnls)
    assert scores2.shape == (N, M)
    assert seg2.shape == (N,)
